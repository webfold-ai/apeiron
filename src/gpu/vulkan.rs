//! The one host cut a GPU has: Vulkan dispatch, hand-rolled — `dlopen("libvulkan.so.1")` + one
//! `vkGetInstanceProcAddr`, no ash/vulkano/crate (the reactor-epoll discipline: the unsafe IS the
//! boundary, not a dependency). Shared by `simt_check` (op gate) and `simt_gemv` (the bandwidth
//! duel). Every struct layout and enum value cited to the local registry
//! `~/Workspace/Vulkan-Docs/xml/vk.xml`, never from memory.
//!
//! Synchronization primitives charted vs CUDA/SPIR-V (timeline semaphores, multi-queue, the
//! device-scope `OpControlBarrier` we already enable but haven't tried natively) — `GPU_PRIMITIVES.md`.

// Raw Vulkan FFI relocated verbatim from the decode harness: every struct member is a vk.xml
// field name (documented at its citation, not per-member here), so `missing_docs` is silenced for
// this boundary — the doc discipline lives on the runtime layer (`super::runtime`) above it.
#![allow(
    dead_code,
    missing_docs,
    unused_parens,
    clippy::too_many_arguments,
    clippy::missing_safety_doc
)]

use std::ffi::c_void;
use std::os::raw::c_char;

// ── dlopen/dlsym (glibc; RTLD_NOW = 2, bits/dlfcn.h) ────────────────────────────────────────────
extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

// ── vk.xml facts ────────────────────────────────────────────────────────────────────────────────
// Dispatchable handles are pointers; non-dispatchable are 64-bit.
pub type Instance = *mut c_void;
pub type PhysicalDevice = *mut c_void;
pub type Device = *mut c_void;
pub type Queue = *mut c_void;
pub type CommandBuffer = *mut c_void;
pub type Handle = u64;

pub const VK_SUCCESS: i32 = 0;
// VK_MAKE_API_VERSION(v, M, m, p) = (v<<29)|(M<<22)|(m<<12)|p — vk.xml macro. 1.1 for SPIR-V 1.3.
pub const API_VERSION_1_1: u32 = 1 << 22 | 1 << 12;
// sType values (vk.xml VkStructureType):
pub const ST_APPLICATION_INFO: u32 = 0;
pub const ST_INSTANCE_CREATE_INFO: u32 = 1;
pub const ST_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const ST_DEVICE_CREATE_INFO: u32 = 3;
pub const ST_SUBMIT_INFO: u32 = 4;
pub const ST_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const ST_BUFFER_CREATE_INFO: u32 = 12;
pub const ST_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const ST_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const ST_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const ST_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const ST_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const ST_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const ST_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const ST_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const ST_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const ST_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const ST_MEMORY_BARRIER: u32 = 46;
pub const ST_FENCE_CREATE_INFO: u32 = 8;
pub const ST_QUERY_POOL_CREATE_INFO: u32 = 11;
pub const VK_TIMEOUT: i32 = 2;
// Timestamp queries (vk.xml): VK_QUERY_TYPE_TIMESTAMP=1; result flags 64_BIT=1, WAIT_BIT=2.
// The write is done via synchronization2 (vkCmdWriteTimestamp2) — the legacy vkCmdWriteTimestamp
// is superseded and, on this Blackwell driver, flips availability but writes 0. The stage flags
// are VkPipelineStageFlags2 (u64): TOP_OF_PIPE=bitpos 0, BOTTOM_OF_PIPE=bitpos 13.
pub const QUERY_TYPE_TIMESTAMP: u32 = 2;
pub const QUERY_RESULT_64: u32 = 1;
pub const QUERY_RESULT_WAIT: u32 = 2;
pub const STAGE_TOP_OF_PIPE: u64 = 1;
pub const STAGE_BOTTOM_OF_PIPE: u64 = 1 << 13;
pub const STAGE_ALL_COMMANDS: u64 = 1 << 16; // sync2 ALL_COMMANDS (TOP/BOTTOM read as NONE here)
                                             // sync2 barrier (VkMemoryBarrier2 / VkDependencyInfo, ext 315). 64-bit stage/access words.
pub const ST_MEMORY_BARRIER_2: u32 = 1000314000;
pub const ST_DEPENDENCY_INFO: u32 = 1000314003;
pub const STAGE2_COMPUTE: u64 = 0x800; // VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
pub const ACCESS2_SHADER_READ: u64 = 0x20; // VK_ACCESS_2_SHADER_READ_BIT
pub const ACCESS2_SHADER_WRITE: u64 = 0x40; // VK_ACCESS_2_SHADER_WRITE_BIT
                                            // VkPhysicalDeviceSynchronization2Features sType (vk.xml ext 315, offset 7 → 1000314007).
const ST_FEATURES_SYNCHRONIZATION2: u32 = 1000314007;
// VkPhysicalDeviceHostQueryResetFeatures sType (vk.xml ext 262, offset 0 → 1000261000).
const ST_FEATURES_HOST_QUERY_RESET: u32 = 1000261000;
// Enums/bits (vk.xml):
pub const QUEUE_COMPUTE_BIT: u32 = 1 << 1;
pub const MEMORY_DEVICE_LOCAL: u32 = 1 << 0;
pub const MEMORY_HOST_VISIBLE: u32 = 1 << 1;
pub const MEMORY_HOST_COHERENT: u32 = 1 << 2;
pub const MEMORY_HOST_CACHED: u32 = 1 << 3; // cached mapping — FAST CPU reads (readback)
pub const BUFFER_USAGE_TRANSFER_SRC: u32 = 1 << 0;
pub const BUFFER_USAGE_TRANSFER_DST: u32 = 1 << 1;
pub const BUFFER_USAGE_STORAGE: u32 = 1 << 5;
/// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` — the buffer's GPU address may be taken.
pub const BUFFER_USAGE_SHADER_DEVICE_ADDRESS: u32 = 1 << 17;
pub const SHARING_EXCLUSIVE: u32 = 0;
pub const SHADER_STAGE_COMPUTE: u32 = 1 << 5;
pub const DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
pub const PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
pub const COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const WHOLE_SIZE: u64 = !0;
pub const STAGE_TRANSFER: u32 = 1 << 12;
pub const STAGE_COMPUTE: u32 = 1 << 11;
pub const STAGE_HOST: u32 = 1 << 14;
pub const ACCESS_SHADER_READ: u32 = 1 << 5;
pub const ACCESS_SHADER_WRITE: u32 = 1 << 6;
pub const ACCESS_TRANSFER_READ: u32 = 1 << 11;
pub const ACCESS_TRANSFER_WRITE: u32 = 1 << 12;
pub const ACCESS_HOST_READ: u32 = 1 << 13;

// ── vk.xml struct layouts (member order verbatim; #[repr(C)] reproduces the C padding) ─────────
#[repr(C)]
pub struct ApplicationInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}
#[repr(C)]
pub struct InstanceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_application_info: *const ApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}
#[repr(C)]
pub struct DeviceQueueCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}
#[repr(C)]
pub struct DeviceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const DeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    pub p_enabled_features: *const c_void,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: Extent3D,
}
#[repr(C)]
pub struct BufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: u64,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}
#[repr(C)]
#[derive(Default)]
pub struct MemoryRequirements {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MemoryHeap {
    pub size: u64,
    pub flags: u32,
}
// VK_MAX_MEMORY_TYPES = 32, VK_MAX_MEMORY_HEAPS = 16 (vk.xml API constants).
#[repr(C)]
pub struct PhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [MemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [MemoryHeap; 16],
}
#[repr(C)]
pub struct MemoryAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub allocation_size: u64,
    pub memory_type_index: u32,
}
#[repr(C)]
pub struct ShaderModuleCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}
#[repr(C)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub p_immutable_samplers: *const c_void,
}
#[repr(C)]
pub struct DescriptorSetLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const DescriptorSetLayoutBinding,
}
#[repr(C)]
pub struct PipelineLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const Handle,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const c_void,
}
#[repr(C)]
pub struct PipelineShaderStageCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: Handle,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}
#[repr(C)]
pub struct ComputePipelineCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: PipelineShaderStageCreateInfo,
    pub layout: Handle,
    pub base_pipeline_handle: Handle,
    pub base_pipeline_index: i32,
}
#[repr(C)]
pub struct DescriptorPoolSize {
    pub ty: u32,
    pub descriptor_count: u32,
}
#[repr(C)]
pub struct DescriptorPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const DescriptorPoolSize,
}
#[repr(C)]
pub struct DescriptorSetAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub descriptor_pool: Handle,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const Handle,
}
#[repr(C)]
pub struct DescriptorBufferInfo {
    pub buffer: Handle,
    pub offset: u64,
    pub range: u64,
}
#[repr(C)]
pub struct WriteDescriptorSet {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub dst_set: Handle,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const c_void,
    pub p_buffer_info: *const DescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}
#[repr(C)]
pub struct CommandPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}
#[repr(C)]
pub struct CommandBufferAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub command_pool: Handle,
    pub level: u32,
    pub command_buffer_count: u32,
}
#[repr(C)]
pub struct CommandBufferBeginInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}
#[repr(C)]
pub struct SubmitInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const Handle,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const CommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const Handle,
}
#[repr(C)]
pub struct MemoryBarrier {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
}
// sync2: stage masks live INSIDE the barrier (64-bit), not as vkCmdPipelineBarrier args.
#[repr(C)]
pub struct MemoryBarrier2 {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_stage_mask: u64,
    pub src_access_mask: u64,
    pub dst_stage_mask: u64,
    pub dst_access_mask: u64,
}
#[repr(C)]
pub struct DependencyInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub dependency_flags: u32,
    pub memory_barrier_count: u32,
    pub p_memory_barriers: *const MemoryBarrier2,
    pub buffer_memory_barrier_count: u32,
    pub p_buffer_memory_barriers: *const c_void,
    pub image_memory_barrier_count: u32,
    pub p_image_memory_barriers: *const c_void,
}
#[repr(C)]
pub struct BufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}
#[repr(C)]
pub struct FenceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}
#[repr(C)]
pub struct QueryPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub query_type: u32,
    pub query_count: u32,
    pub pipeline_statistics: u32,
}
// VkPhysicalDeviceLimits — the full member list in vk.xml order (repr(C) reproduces the C offsets,
// so `timestamp_period` lands where the driver wrote it instead of a hand-counted byte offset).
// Contains VkDeviceSize (u64) / size_t members ⇒ struct alignment 8.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PhysicalDeviceLimits {
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias: f32,
    pub max_sampler_anisotropy: f32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub viewport_bounds_range: [f32; 2],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset: f32,
    pub max_interpolation_offset: f32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: u32,
    pub framebuffer_depth_sample_counts: u32,
    pub framebuffer_stencil_sample_counts: u32,
    pub framebuffer_no_attachments_sample_counts: u32,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: u32,
    pub sampled_image_integer_sample_counts: u32,
    pub sampled_image_depth_sample_counts: u32,
    pub sampled_image_stencil_sample_counts: u32,
    pub storage_image_sample_counts: u32,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: u32,
    pub timestamp_period: f32, // ns per timestamp tick — the target field
                               // (remaining limits members omitted: they follow timestamp_period, so its offset is fixed)
}
// VkPhysicalDeviceProperties up to `limits` (sparseProperties trails it). repr(C) pads the 292-byte
// header (5×u32 + name[256] + uuid[16]) to the 8-aligned start of the limits sub-struct for us.
#[repr(C)]
pub struct PhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [c_char; 256],
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: PhysicalDeviceLimits,
}

// ── function pointer types (C ABI, linux/x86_64) ────────────────────────────────────────────────
pub type GetProcAddr = unsafe extern "C" fn(Instance, *const c_char) -> *const c_void;
pub type GetDeviceProcAddr = unsafe extern "C" fn(Device, *const c_char) -> *const c_void;
type CreateInstance =
    unsafe extern "C" fn(*const InstanceCreateInfo, *const c_void, *mut Instance) -> i32;
type EnumeratePhysicalDevices =
    unsafe extern "C" fn(Instance, *mut u32, *mut PhysicalDevice) -> i32;
type GetQueueFamilyProps =
    unsafe extern "C" fn(PhysicalDevice, *mut u32, *mut QueueFamilyProperties);
type GetMemoryProps = unsafe extern "C" fn(PhysicalDevice, *mut PhysicalDeviceMemoryProperties);
type GetDeviceProps = unsafe extern "C" fn(PhysicalDevice, *mut u8);
type GetDeviceFeatures = unsafe extern "C" fn(PhysicalDevice, *mut PhysicalDeviceFeatures);
type CreateDevice = unsafe extern "C" fn(
    PhysicalDevice,
    *const DeviceCreateInfo,
    *const c_void,
    *mut Device,
) -> i32;
type GetDeviceQueue = unsafe extern "C" fn(Device, u32, u32, *mut Queue);
type CreateBuffer =
    unsafe extern "C" fn(Device, *const BufferCreateInfo, *const c_void, *mut Handle) -> i32;
type GetBufferMemReq = unsafe extern "C" fn(Device, Handle, *mut MemoryRequirements);
type AllocateMemory =
    unsafe extern "C" fn(Device, *const MemoryAllocateInfo, *const c_void, *mut Handle) -> i32;
type BindBufferMemory = unsafe extern "C" fn(Device, Handle, Handle, u64) -> i32;
type MapMemory = unsafe extern "C" fn(Device, Handle, u64, u64, u32, *mut *mut c_void) -> i32;
type CreateShaderModule =
    unsafe extern "C" fn(Device, *const ShaderModuleCreateInfo, *const c_void, *mut Handle) -> i32;
type CreateDescriptorSetLayout = unsafe extern "C" fn(
    Device,
    *const DescriptorSetLayoutCreateInfo,
    *const c_void,
    *mut Handle,
) -> i32;
type CreatePipelineLayout = unsafe extern "C" fn(
    Device,
    *const PipelineLayoutCreateInfo,
    *const c_void,
    *mut Handle,
) -> i32;
type CreateComputePipelines = unsafe extern "C" fn(
    Device,
    Handle,
    u32,
    *const ComputePipelineCreateInfo,
    *const c_void,
    *mut Handle,
) -> i32;
type CreateDescriptorPool = unsafe extern "C" fn(
    Device,
    *const DescriptorPoolCreateInfo,
    *const c_void,
    *mut Handle,
) -> i32;
type AllocateDescriptorSets =
    unsafe extern "C" fn(Device, *const DescriptorSetAllocateInfo, *mut Handle) -> i32;
type UpdateDescriptorSets =
    unsafe extern "C" fn(Device, u32, *const WriteDescriptorSet, u32, *const c_void);
type CreateCommandPool =
    unsafe extern "C" fn(Device, *const CommandPoolCreateInfo, *const c_void, *mut Handle) -> i32;
type AllocateCommandBuffers =
    unsafe extern "C" fn(Device, *const CommandBufferAllocateInfo, *mut CommandBuffer) -> i32;
type BeginCommandBuffer = unsafe extern "C" fn(CommandBuffer, *const CommandBufferBeginInfo) -> i32;
type CmdBindPipeline = unsafe extern "C" fn(CommandBuffer, u32, Handle);
type CmdBindDescriptorSets =
    unsafe extern "C" fn(CommandBuffer, u32, Handle, u32, u32, *const Handle, u32, *const u32);
type CmdDispatch = unsafe extern "C" fn(CommandBuffer, u32, u32, u32);
type CmdCopyBuffer = unsafe extern "C" fn(CommandBuffer, Handle, Handle, u32, *const BufferCopy);
type CmdFillBuffer = unsafe extern "C" fn(CommandBuffer, Handle, u64, u64, u32);
type CmdPipelineBarrier = unsafe extern "C" fn(
    CommandBuffer,
    u32,
    u32,
    u32,
    u32,
    *const MemoryBarrier,
    u32,
    *const c_void,
    u32,
    *const c_void,
);
type EndCommandBuffer = unsafe extern "C" fn(CommandBuffer) -> i32;
type ResetCommandBuffer = unsafe extern "C" fn(CommandBuffer, u32) -> i32;
type QueueSubmit = unsafe extern "C" fn(Queue, u32, *const SubmitInfo, Handle) -> i32;
type QueueWaitIdle = unsafe extern "C" fn(Queue) -> i32;
type CreateFence =
    unsafe extern "C" fn(Device, *const FenceCreateInfo, *const c_void, *mut Handle) -> i32;
type WaitForFences = unsafe extern "C" fn(Device, u32, *const Handle, u32, u64) -> i32;
type ResetFences = unsafe extern "C" fn(Device, u32, *const Handle) -> i32;
type DestroyFence = unsafe extern "C" fn(Device, Handle, *const c_void);
// timestamp query pool: create, reset-in-cb, write-in-cb, read-back (vk.xml core 1.0 signatures)
type CreateQueryPool =
    unsafe extern "C" fn(Device, *const QueryPoolCreateInfo, *const c_void, *mut Handle) -> i32;
type CmdResetQueryPool = unsafe extern "C" fn(CommandBuffer, Handle, u32, u32);
// vkCmdWriteTimestamp2: VkPipelineStageFlags2 stage is a 64-bit flag word (sync2)
type CmdWriteTimestamp2 = unsafe extern "C" fn(CommandBuffer, u64, Handle, u32);
// vkCmdPipelineBarrier2: the whole dependency is one struct (stages inside the barriers)
type CmdPipelineBarrier2 = unsafe extern "C" fn(CommandBuffer, *const DependencyInfo);
type ResetQueryPool = unsafe extern "C" fn(Device, Handle, u32, u32);
type GetQueryPoolResults =
    unsafe extern "C" fn(Device, Handle, u32, u32, usize, *mut c_void, u64, u32) -> i32;

// ── VK_KHR_pipeline_executable_properties (ext 270): the SASS/IR + stats readout ────────────────
// The floor Vulkan will hand us below SPIR-V — the driver's own compiled representation of our
// pipeline (on NVIDIA, the SASS-shaped internal IR) plus per-executable statistics (registers,
// occupancy). All values cited to vk.xml: ext number 270 → sType base 1000269000.
const ST_PIPELINE_INFO: u32 = 1000269001;
const ST_PIPELINE_EXECUTABLE_PROPERTIES: u32 = 1000269002;
const ST_PIPELINE_EXECUTABLE_INFO: u32 = 1000269003;
const ST_PIPELINE_EXECUTABLE_STATISTIC: u32 = 1000269004;
const ST_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION: u32 = 1000269005;
const ST_FEATURES_PIPELINE_EXECUTABLE: u32 = 1000269000;
const ST_FEATURES_VULKAN_MEMORY_MODEL: u32 = 1000211000;
#[repr(C)]
struct PhysDeviceVulkanMemoryModelFeatures {
    s_type: u32,
    p_next: *mut c_void,
    vulkan_memory_model: u32,
    vulkan_memory_model_device_scope: u32,
    vulkan_memory_model_availability_visibility_chains: u32,
}
#[repr(C)]
struct PhysDevice16BitStorageFeatures {
    s_type: u32,
    p_next: *mut c_void,
    storage_buffer_16bit_access: u32,
    uniform_and_storage_buffer_16bit_access: u32,
    storage_push_constant16: u32,
    storage_input_output16: u32,
}
const ST_FEATURES_16BIT_STORAGE: u32 = 1000083000;
#[repr(C)]
struct PhysDeviceCooperativeMatrixFeatures {
    s_type: u32,
    p_next: *mut c_void,
    cooperative_matrix: u32,
    cooperative_matrix_robust_buffer_access: u32,
}
const ST_FEATURES_COOPERATIVE_MATRIX: u32 = 1000506000;
#[repr(C)]
#[derive(Clone, Copy)]
struct CooperativeMatrixPropertiesKHR {
    s_type: u32,
    p_next: *mut c_void,
    m_size: u32,
    n_size: u32,
    k_size: u32,
    a_type: u32,
    b_type: u32,
    c_type: u32,
    result_type: u32,
    saturating_accumulation: u32,
    scope: u32,
}
const ST_COOPERATIVE_MATRIX_PROPERTIES: u32 = 1000506001;
const VK_COMPONENT_TYPE_SINT8_KHR: u32 = 3;
const VK_COMPONENT_TYPE_SINT32_KHR: u32 = 5;
const VK_SCOPE_SUBGROUP_KHR: u32 = 3;
type GetCooperativeMatrixPropertiesKHR =
    unsafe extern "C" fn(PhysicalDevice, *mut u32, *mut CooperativeMatrixPropertiesKHR) -> i32;
#[repr(C)]
struct PhysDevice8BitStorageFeatures {
    s_type: u32,
    p_next: *mut c_void,
    storage_buffer_8bit_access: u32,
    uniform_and_storage_buffer_8bit_access: u32,
    storage_push_constant8: u32,
}
const ST_FEATURES_8BIT_STORAGE: u32 = 1000177000;
#[repr(C)]
struct PhysDeviceShaderFloat16Int8Features {
    s_type: u32,
    p_next: *mut c_void,
    shader_float16: u32,
    shader_int8: u32,
}
const ST_FEATURES_SHADER_FLOAT16_INT8: u32 = 1000082000;
/// `VK_KHR_buffer_device_address` — the feature that makes a buffer's GPU address readable, so a
/// shader can dereference it directly (SPIR-V `PhysicalStorageBuffer`) instead of going through a
/// descriptor. That is what removes the driver's per-load descriptor-range compare.
#[repr(C)]
struct PhysDeviceBufferDeviceAddressFeatures {
    s_type: u32,
    p_next: *mut c_void,
    buffer_device_address: u32,
    buffer_device_address_capture_replay: u32,
    buffer_device_address_multi_device: u32,
}
const ST_FEATURES_BUFFER_DEVICE_ADDRESS: u32 = 1000257000;
/// `VkPhysicalDeviceFeatures` as its 55 `VkBool32`s. Only ONE is ever set here — `shaderInt64`
/// (index 40), which gates SPIR-V's `Int64` capability, which `OpConvertUToPtr` needs to turn a
/// 64-bit device address into a pointer. **`robustBufferAccess` is index 0 and stays 0**: the whole
/// point is that the bound is the mandala mask, not a driver range check.
#[repr(C)]
#[derive(Clone, Copy)]
struct PhysicalDeviceFeatures {
    bits: [u32; 55],
}
impl Default for PhysicalDeviceFeatures {
    fn default() -> Self {
        PhysicalDeviceFeatures { bits: [0; 55] }
    }
}
const FEATURE_SHADER_FLOAT64: usize = 39;
const FEATURE_SHADER_INT64: usize = 40;
const FEATURE_ROBUST_BUFFER_ACCESS: usize = 0;
/// `VkPhysicalDeviceFeatures2`, so the core feature list travels in the same `pNext` chain as the
/// extension features (the alternative, `p_enabled_features`, cannot coexist with it).
#[repr(C)]
struct PhysDeviceFeatures2 {
    s_type: u32,
    p_next: *mut c_void,
    features: PhysicalDeviceFeatures,
}
const ST_FEATURES_2: u32 = 1000059000;
/// `VkMemoryAllocateFlagsInfo` — an allocation backing a `SHADER_DEVICE_ADDRESS` buffer must say so.
#[repr(C)]
struct MemoryAllocateFlagsInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    device_mask: u32,
}
const ST_MEMORY_ALLOCATE_FLAGS_INFO: u32 = 1000060000;
const MEMORY_ALLOCATE_DEVICE_ADDRESS: u32 = 1 << 1;
/// `VkImportMemoryHostPointerInfoEXT` — the p_next that makes an allocation ADOPT memory the host
/// already owns, instead of the driver allocating its own.
///
/// **This is what puts the KV on DISK rather than in RAM.** `vkAllocateMemory(HOST_VISIBLE)` hands
/// back anonymous driver-owned pages: not file-backed, effectively unpageable, so under pressure the
/// kernel cannot evict them — it pushes everything else into swap and then OOM-kills. Importing an
/// `mmap(MAP_SHARED)` of a file instead makes the same buffer PAGE-CACHE backed, and the kernel
/// pages cold blocks out to the SSD on demand while the device keeps reading them.
///
/// MEASURED on this box (`vulkaninfo`): `minImportedHostPointerAlignment = 0x1000` — one page, which
/// `mmap` satisfies by construction — and `maxMemoryAllocationSize` is unbounded.
#[repr(C)]
struct ImportMemoryHostPointerInfo {
    s_type: u32,
    p_next: *const c_void,
    handle_type: u32,
    p_host_pointer: *mut c_void,
}
const ST_IMPORT_MEMORY_HOST_POINTER_INFO: u32 = 1000178000;

// `mmap`/`munmap`, declared rather than pulled in as a dependency: this file is a hand-written
// Vulkan loader and already speaks raw FFI, and two syscalls do not justify a new crate edge in
// `modelfold`. Signatures from `mman.h`; the constants are the Linux ABI's and are stable.
unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, length: usize) -> i32;
}
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_SHARED: i32 = 1;
const MAP_SHARED_ANON: i32 = 1 | 0x20; // MAP_SHARED | MAP_ANONYMOUS
const ST_MEMORY_HOST_POINTER_PROPERTIES: u32 = 1000178001;
const EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION: u32 = 1 << 7;

/// `VkMemoryHostPointerPropertiesEXT` — which memory types may back an imported pointer. The driver
/// decides this, so it is ASKED rather than assumed: picking a type outside this mask is a
/// validation error and, on a release build, undefined behaviour.
#[repr(C)]
struct MemoryHostPointerProperties {
    s_type: u32,
    p_next: *mut c_void,
    memory_type_bits: u32,
}
type GetMemoryHostPointerProperties =
    unsafe extern "C" fn(Device, u32, *const c_void, *mut MemoryHostPointerProperties) -> i32;

/// `VkBufferDeviceAddressInfo`.
#[repr(C)]
struct BufferDeviceAddressInfo {
    s_type: u32,
    p_next: *const c_void,
    buffer: Handle,
}
const ST_BUFFER_DEVICE_ADDRESS_INFO: u32 = 1000244001;
type GetBufferDeviceAddress = unsafe extern "C" fn(Device, *const BufferDeviceAddressInfo) -> u64;

#[repr(C)]
struct PhysDeviceSynchronization2Features {
    s_type: u32,
    p_next: *mut c_void,
    synchronization2: u32, // VkBool32
}
#[repr(C)]
struct PhysDeviceHostQueryResetFeatures {
    s_type: u32,
    p_next: *mut c_void,
    host_query_reset: u32, // VkBool32
}
// VkPipelineCreateFlagBits: capture bits 6/7 (vk.xml).
pub const PIPELINE_CREATE_CAPTURE_STATISTICS: u32 = 1 << 6;
pub const PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS: u32 = 1 << 7;
const DESC_SIZE: usize = 256; // VK_MAX_DESCRIPTION_SIZE

#[repr(C)]
struct PhysDevicePipelineExecutableFeatures {
    s_type: u32,
    p_next: *mut c_void,
    pipeline_executable_info: u32, // VkBool32
}
#[repr(C)]
struct ExtensionProperties {
    extension_name: [c_char; 256],
    spec_version: u32,
}
#[repr(C)]
struct PipelineInfoKHR {
    s_type: u32,
    p_next: *const c_void,
    pipeline: Handle,
}
#[repr(C)]
struct PipelineExecutablePropertiesKHR {
    s_type: u32,
    p_next: *mut c_void,
    stages: u32,
    name: [c_char; DESC_SIZE],
    description: [c_char; DESC_SIZE],
    subgroup_size: u32,
}
#[repr(C)]
struct PipelineExecutableInfoKHR {
    s_type: u32,
    p_next: *const c_void,
    pipeline: Handle,
    executable_index: u32,
}
#[repr(C)]
struct PipelineExecutableInternalRepresentationKHR {
    s_type: u32,
    p_next: *mut c_void,
    name: [c_char; DESC_SIZE],
    description: [c_char; DESC_SIZE],
    is_text: u32,
    data_size: usize,
    p_data: *mut c_void,
}
#[repr(C)]
struct PipelineExecutableStatisticKHR {
    s_type: u32,
    p_next: *mut c_void,
    name: [c_char; DESC_SIZE],
    description: [c_char; DESC_SIZE],
    format: u32,
    // union { VkBool32; int64; uint64; double } — 8 bytes, aligned 8
    value: u64,
}

const ST_PIPELINE_CACHE_CREATE_INFO: u32 = 17;
#[repr(C)]
struct PipelineCacheCreateInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    initial_data_size: usize,
    p_initial_data: *const c_void,
}
type CreatePipelineCache =
    unsafe extern "C" fn(Device, *const PipelineCacheCreateInfo, *const c_void, *mut Handle) -> i32;
type GetPipelineCacheData = unsafe extern "C" fn(Device, Handle, *mut usize, *mut c_void) -> i32;

type EnumDeviceExtProps =
    unsafe extern "C" fn(PhysicalDevice, *const c_char, *mut u32, *mut ExtensionProperties) -> i32;
type GetExecProps = unsafe extern "C" fn(
    Device,
    *const PipelineInfoKHR,
    *mut u32,
    *mut PipelineExecutablePropertiesKHR,
) -> i32;
type GetExecIR = unsafe extern "C" fn(
    Device,
    *const PipelineExecutableInfoKHR,
    *mut u32,
    *mut PipelineExecutableInternalRepresentationKHR,
) -> i32;
type GetExecStats = unsafe extern "C" fn(
    Device,
    *const PipelineExecutableInfoKHR,
    *mut u32,
    *mut PipelineExecutableStatisticKHR,
) -> i32;

fn cstr(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// One driver statistic (register count, occupancy, …) as name + value.
pub struct Stat {
    pub name: String,
    pub value: String,
}

/// A timestamp query pool + its query count. `count` queries, read back as u64 ticks.
pub struct QueryPool {
    pub handle: Handle,
    pub count: u32,
}

// ── NVML: the energy counter (libnvidia-ml). `nvmlDeviceGetTotalEnergyConsumption` is a
//    monotonic millijoule counter — read before/after a timed region for real joules. ───────────
pub mod nvml {
    use std::os::raw::{c_char, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: i32 = 2;
    type Device = *mut c_void;
    type Init = unsafe extern "C" fn() -> i32;
    type GetHandle = unsafe extern "C" fn(u32, *mut Device) -> i32;
    type GetEnergy = unsafe extern "C" fn(Device, *mut u64) -> i32;
    type GetPower = unsafe extern "C" fn(Device, *mut u32) -> i32;

    /// The energy meter — a handle to GPU 0 plus the two counters.
    pub struct Nvml {
        dev: Device,
        get_energy: Option<GetEnergy>,
        get_power: GetPower,
    }
    // the handle is a driver-owned opaque pointer used read-only from one thread; safe to share
    unsafe impl Sync for Nvml {}
    unsafe impl Send for Nvml {}

    impl Nvml {
        /// Load NVML, init, grab GPU 0. `None` = not present (honest skip).
        pub unsafe fn load() -> Option<Nvml> {
            let lib = dlopen(b"libnvidia-ml.so.1\0".as_ptr().cast::<c_char>(), RTLD_NOW);
            if lib.is_null() {
                return None;
            }
            let sym = |n: &[u8]| dlsym(lib, n.as_ptr().cast::<c_char>());
            let init_ptr = sym(b"nvmlInit_v2\0");
            let handle_ptr = sym(b"nvmlDeviceGetHandleByIndex_v2\0");
            let energy_ptr = sym(b"nvmlDeviceGetTotalEnergyConsumption\0");
            let power_ptr = sym(b"nvmlDeviceGetPowerUsage\0");
            if init_ptr.is_null() || handle_ptr.is_null() || power_ptr.is_null() {
                return None;
            }
            let init: Init = std::mem::transmute(init_ptr);
            let get_handle: GetHandle = std::mem::transmute(handle_ptr);
            if init() != 0 {
                return None;
            }
            let mut dev: Device = std::ptr::null_mut();
            if get_handle(0, &mut dev) != 0 {
                return None;
            }
            Some(Nvml {
                dev,
                get_energy: if energy_ptr.is_null() {
                    None
                } else {
                    Some(std::mem::transmute(energy_ptr))
                },
                get_power: std::mem::transmute(power_ptr),
            })
        }

        /// Total energy in millijoules (monotonic). `None` if the counter is unsupported.
        pub unsafe fn energy_mj(&self) -> Option<u64> {
            let f = self.get_energy?;
            let mut e = 0u64;
            if f(self.dev, &mut e) == 0 {
                Some(e)
            } else {
                None
            }
        }

        /// Instantaneous board power draw in milliwatts.
        pub unsafe fn power_mw(&self) -> Option<u32> {
            let mut p = 0u32;
            if (self.get_power)(self.dev, &mut p) == 0 {
                Some(p)
            } else {
                None
            }
        }
    }
}

pub fn check(r: i32, what: &str) {
    assert!(r == VK_SUCCESS, "{what} failed: VkResult {r}");
}

unsafe fn load_fn(gipa: GetProcAddr, instance: Instance, name: &str) -> *const c_void {
    let c = std::ffi::CString::new(name).unwrap();
    let p = gipa(instance, c.as_ptr());
    assert!(!p.is_null(), "{name} not found in the Vulkan loader");
    p
}

macro_rules! vk_fn {
    ($gipa:expr, $inst:expr, $name:literal, $ty:ty) => {
        std::mem::transmute::<*const c_void, $ty>(load_fn($gipa, $inst, $name))
    };
}

/// The loaded host: instance → device → queue, all entry points resolved, memory map cached.
pub struct Vk {
    pub instance: Instance,
    pub pd: PhysicalDevice,
    pub device: Device,
    pub queue: Queue,
    pub family: u32,
    pub device_name: String,
    /// `maxComputeSharedMemorySize` — the ceiling a workgroup may declare. See `max_shared`'s read.
    pub max_shared: u32,
    pub mem: PhysicalDeviceMemoryProperties,
    // entry points
    pub create_buffer: CreateBuffer,
    pub buf_req: GetBufferMemReq,
    pub alloc_mem: AllocateMemory,
    pub bind_buf: BindBufferMemory,
    pub map_mem: MapMemory,
    pub create_shader: CreateShaderModule,
    pub create_dsl: CreateDescriptorSetLayout,
    pub create_pl: CreatePipelineLayout,
    pub create_pipes: CreateComputePipelines,
    pub create_dpool: CreateDescriptorPool,
    pub alloc_dsets: AllocateDescriptorSets,
    pub update_dsets: UpdateDescriptorSets,
    pub create_cpool: CreateCommandPool,
    pub alloc_cbufs: AllocateCommandBuffers,
    pub begin_cb: BeginCommandBuffer,
    pub bind_pipe: CmdBindPipeline,
    pub bind_sets: CmdBindDescriptorSets,
    pub dispatch: CmdDispatch,
    pub copy_buffer: CmdCopyBuffer,
    pub fill_buffer: CmdFillBuffer,
    pub pipeline_barrier: CmdPipelineBarrier,
    pub end_cb: EndCommandBuffer,
    pub reset_cb: ResetCommandBuffer,
    pub submit: QueueSubmit,
    pub wait_idle: QueueWaitIdle,
    pub create_fence: CreateFence,
    pub wait_fences: WaitForFences,
    pub reset_fences: ResetFences,
    pub destroy_fence: DestroyFence,
    // GPU timestamps: the tick period (ns) and the compute queue's valid-bit width (0 = unsupported)
    pub timestamp_period: f32,
    pub timestamp_valid_bits: u32,
    create_query_pool: CreateQueryPool,
    cmd_reset_query_pool: CmdResetQueryPool,
    reset_query_pool_host: Option<ResetQueryPool>,
    cmd_write_timestamp2_f: CmdWriteTimestamp2,
    cmd_pipeline_barrier2_f: CmdPipelineBarrier2,
    get_query_results: GetQueryPoolResults,
    // pipeline-executable readout (None when the extension is absent)
    pub has_exec: bool,
    get_exec_props: Option<GetExecProps>,
    get_exec_ir: Option<GetExecIR>,
    get_exec_stats: Option<GetExecStats>,
    // the pipeline cache: NVIDIA embeds its compiled cubin here — the real SASS door
    pub cache: Handle,
    get_cache_data: GetPipelineCacheData,
    create_cache: CreatePipelineCache,
    /// `VK_KHR_buffer_device_address` present AND enabled — every buffer then carries the usage bit
    /// and its GPU address can be read with [`Vk::device_address`].
    pub has_bda: bool,
    get_bda: Option<GetBufferDeviceAddress>,
    /// `VK_EXT_external_memory_host` — present and enabled. Gates [`Vk::buffer_from_host_ptr`].
    pub has_exthost: bool,
    get_hostptr_props: Option<GetMemoryHostPointerProperties>,
    /// `VK_KHR_cooperative_matrix` present and int8 16x16x32 shape supported, unless disabled via
    /// `APEIRON_COOPMAT=0`.
    pub has_coopmat: bool,
    /// `shaderFloat64` supported and enabled (only on the `has_bda` feature chain).
    pub has_f64: bool,
}

impl Vk {
    /// Load the loader, create an instance/device on the first compute-capable physical device.
    /// `Err(msg)` = honest skip (no loader / no device), never a hard failure.
    pub unsafe fn load() -> Result<Vk, String> {
        let lib = dlopen(b"libvulkan.so.1\0".as_ptr().cast::<c_char>(), RTLD_NOW);
        if lib.is_null() {
            return Err("libvulkan.so.1 not found".into());
        }
        let gipa_ptr = dlsym(lib, b"vkGetInstanceProcAddr\0".as_ptr().cast::<c_char>());
        if gipa_ptr.is_null() {
            return Err("vkGetInstanceProcAddr not found".into());
        }
        let gipa: GetProcAddr = std::mem::transmute(gipa_ptr);

        let app = ApplicationInfo {
            s_type: ST_APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: b"hashtrinity-simt\0".as_ptr().cast::<c_char>(),
            application_version: 0,
            p_engine_name: std::ptr::null(),
            engine_version: 0,
            api_version: API_VERSION_1_1,
        };
        let ici = InstanceCreateInfo {
            s_type: ST_INSTANCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_application_info: &app,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: std::ptr::null(),
        };
        let create_instance = vk_fn!(
            gipa,
            std::ptr::null_mut(),
            "vkCreateInstance",
            CreateInstance
        );
        let mut instance: Instance = std::ptr::null_mut();
        if create_instance(&ici, std::ptr::null(), &mut instance) != VK_SUCCESS {
            return Err("vkCreateInstance failed (no ICD?)".into());
        }

        let enum_pd = vk_fn!(
            gipa,
            instance,
            "vkEnumeratePhysicalDevices",
            EnumeratePhysicalDevices
        );
        let qfp = vk_fn!(
            gipa,
            instance,
            "vkGetPhysicalDeviceQueueFamilyProperties",
            GetQueueFamilyProps
        );
        let mem_props_f = vk_fn!(
            gipa,
            instance,
            "vkGetPhysicalDeviceMemoryProperties",
            GetMemoryProps
        );
        let dev_props_f = vk_fn!(
            gipa,
            instance,
            "vkGetPhysicalDeviceProperties",
            GetDeviceProps
        );
        let dev_feats_f = vk_fn!(
            gipa,
            instance,
            "vkGetPhysicalDeviceFeatures",
            GetDeviceFeatures
        );
        let create_device = vk_fn!(gipa, instance, "vkCreateDevice", CreateDevice);

        let mut n = 0u32;
        check(
            enum_pd(instance, &mut n, std::ptr::null_mut()),
            "enumerate count",
        );
        if n == 0 {
            return Err("no Vulkan physical device".into());
        }
        let mut pds = vec![std::ptr::null_mut(); n as usize];
        check(
            enum_pd(instance, &mut n, pds.as_mut_ptr()),
            "enumerate devices",
        );

        let mut picked: Option<(PhysicalDevice, u32, u32)> = None;
        for &pd in &pds {
            let mut qn = 0u32;
            qfp(pd, &mut qn, std::ptr::null_mut());
            let zero = Extent3D {
                width: 0,
                height: 0,
                depth: 0,
            };
            let mut fams = vec![
                QueueFamilyProperties {
                    queue_flags: 0,
                    queue_count: 0,
                    timestamp_valid_bits: 0,
                    min_image_transfer_granularity: zero
                };
                qn as usize
            ];
            qfp(pd, &mut qn, fams.as_mut_ptr());
            if let Some(i) = fams
                .iter()
                .position(|f| f.queue_flags & QUEUE_COMPUTE_BIT != 0)
            {
                picked = Some((pd, i as u32, fams[i].timestamp_valid_bits));
                break;
            }
        }
        let Some((pd, family, timestamp_valid_bits)) = picked else {
            return Err("no compute queue family".into());
        };

        // VkPhysicalDeviceProperties: 5 × u32, then deviceName[256] (vk.xml member order); the
        // whole struct (limits included) is well under the 4 KiB scratch receiving it.
        let mut props = [0u8; 4096];
        dev_props_f(pd, props.as_mut_ptr());
        let name_bytes = &props[20..20 + 256];
        let name_len = name_bytes.iter().position(|&b| b == 0).unwrap_or(0);
        let device_name = String::from_utf8_lossy(&name_bytes[..name_len]).into_owned();
        // the timestamp tick period (ns): read via the typed layout so the compiler owns the offset
        let timestamp_period = (*(props.as_ptr() as *const PhysicalDeviceProperties))
            .limits
            .timestamp_period;
        // **The shared-memory ceiling a workgroup may declare.** Read from the DEVICE rather than
        // assumed, because it is the discriminator the gemv mapping is chosen on
        // (`results/gemv-shared-is-over-the-limit.md`): the row map stages the activation in shared
        // at 5.25 B per k element, so a long-k projection asks for more than this and the driver
        // accepts it without complaint. 49152 on this RTX 5080.
        let max_shared = (*(props.as_ptr() as *const PhysicalDeviceProperties))
            .limits
            .max_compute_shared_memory_size;

        // Does this device expose the pipeline-executable readout? Enumerate its extensions.
        let enum_dext = vk_fn!(
            gipa,
            instance,
            "vkEnumerateDeviceExtensionProperties",
            EnumDeviceExtProps
        );
        let mut en = 0u32;
        enum_dext(pd, std::ptr::null(), &mut en, std::ptr::null_mut());
        let mut exts = (0..en)
            .map(|_| ExtensionProperties {
                extension_name: [0; 256],
                spec_version: 0,
            })
            .collect::<Vec<_>>();
        enum_dext(pd, std::ptr::null(), &mut en, exts.as_mut_ptr());
        if std::env::var("DEC_LISTEXT").is_ok() {
            let mut names: Vec<String> = exts
                .iter()
                .map(|e| cstr(&e.extension_name).to_string())
                .collect();
            names.sort();
            for n in &names {
                println!("EXT: {n}");
            }
            println!("EXT_TOTAL: {}", names.len());
        }
        let has_exec = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_pipeline_executable_properties");
        let has_mm = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_vulkan_memory_model");
        let has_sync2 = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_synchronization2");
        let has_hqr = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_EXT_host_query_reset");
        let has_coopmat_ext = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_cooperative_matrix");
        let mut has_coopmat_shape = false;
        if has_coopmat_ext {
            let c = std::ffi::CStr::from_bytes_with_nul(
                b"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\0",
            )
            .unwrap();
            let p = gipa(instance, c.as_ptr());
            if !p.is_null() {
                let get_coopmat_props: GetCooperativeMatrixPropertiesKHR = std::mem::transmute(p);
                let mut count = 0u32;
                let r = get_coopmat_props(pd, &mut count, std::ptr::null_mut());
                if r == 0 && count > 0 {
                    let mut props = vec![
                        CooperativeMatrixPropertiesKHR {
                            s_type: ST_COOPERATIVE_MATRIX_PROPERTIES,
                            p_next: std::ptr::null_mut(),
                            m_size: 0,
                            n_size: 0,
                            k_size: 0,
                            a_type: 0,
                            b_type: 0,
                            c_type: 0,
                            result_type: 0,
                            saturating_accumulation: 0,
                            scope: 0,
                        };
                        count as usize
                    ];
                    let r = get_coopmat_props(pd, &mut count, props.as_mut_ptr());
                    if r == 0 {
                        has_coopmat_shape = props.iter().any(|prop| {
                            prop.m_size == 16
                                && prop.n_size == 16
                                && prop.k_size == 32
                                && prop.a_type == VK_COMPONENT_TYPE_SINT8_KHR
                                && prop.b_type == VK_COMPONENT_TYPE_SINT8_KHR
                                && prop.c_type == VK_COMPONENT_TYPE_SINT32_KHR
                                && prop.result_type == VK_COMPONENT_TYPE_SINT32_KHR
                                && prop.scope == VK_SCOPE_SUBGROUP_KHR
                        });
                    }
                }
            }
        }
        let coopmat_allowed = !std::env::var("APEIRON_COOPMAT").map_or(false, |v| v.trim() == "0");
        let has_coopmat = has_coopmat_ext && has_coopmat_shape && coopmat_allowed;
        let has_8bit = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_8bit_storage");
        let has_shaderint8 = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_shader_float16_int8");
        // the mandala lever: a raw GPU address in place of a descriptor. `SANG_NO_BDA=1` forces the
        // descriptor path back on, so the A/B is one env var and not a rebuild.
        let has_bda = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_buffer_device_address")
            && std::env::var_os("SANG_NO_BDA").is_none();
        // the DISK tier's mechanism: import an mmap'd file as device-readable memory
        let has_exthost = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_EXT_external_memory_host");

        let prio = 1.0f32;
        let qci = DeviceQueueCreateInfo {
            s_type: ST_DEVICE_QUEUE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: family,
            queue_count: 1,
            p_queue_priorities: &prio,
        };
        // enable the extension + its feature bit when present (the pNext feature chain is how the
        // driver is told to actually honour the CAPTURE pipeline flags)
        let feat = PhysDevicePipelineExecutableFeatures {
            s_type: ST_FEATURES_PIPELINE_EXECUTABLE,
            p_next: std::ptr::null_mut(),
            pipeline_executable_info: 1,
        };
        // the grid-sync memory model — device-scope atomics with AV chains (the megakernel's
        // cross-workgroup hand-off); chained ahead of the pipeline-exec feature
        let mm_feat = PhysDeviceVulkanMemoryModelFeatures {
            s_type: ST_FEATURES_VULKAN_MEMORY_MODEL,
            p_next: if has_exec {
                (&feat as *const PhysDevicePipelineExecutableFeatures as *mut c_void)
            } else {
                std::ptr::null_mut()
            },
            vulkan_memory_model: 1,
            vulkan_memory_model_device_scope: 1,
            vulkan_memory_model_availability_visibility_chains: 1,
        };
        let has_16bit = exts
            .iter()
            .any(|e| cstr(&e.extension_name) == "VK_KHR_16bit_storage");
        let mut ext_ptrs: Vec<*const c_char> = Vec::new();
        if has_exec {
            ext_ptrs.push(
                b"VK_KHR_pipeline_executable_properties\0"
                    .as_ptr()
                    .cast::<c_char>(),
            );
        }
        if has_mm {
            ext_ptrs.push(b"VK_KHR_vulkan_memory_model\0".as_ptr().cast::<c_char>());
        }
        if has_16bit {
            ext_ptrs.push(b"VK_KHR_16bit_storage\0".as_ptr().cast::<c_char>());
        }
        if has_sync2 {
            ext_ptrs.push(b"VK_KHR_synchronization2\0".as_ptr().cast::<c_char>());
        }
        if has_hqr {
            ext_ptrs.push(b"VK_EXT_host_query_reset\0".as_ptr().cast::<c_char>());
        }
        if has_coopmat {
            ext_ptrs.push(b"VK_KHR_cooperative_matrix\0".as_ptr().cast::<c_char>());
        }
        if has_8bit {
            ext_ptrs.push(b"VK_KHR_8bit_storage\0".as_ptr().cast::<c_char>());
        }
        if has_bda {
            ext_ptrs.push(b"VK_KHR_buffer_device_address\0".as_ptr().cast::<c_char>());
        }
        if has_shaderint8 {
            ext_ptrs.push(b"VK_KHR_shader_float16_int8\0".as_ptr().cast::<c_char>());
        }
        if has_exthost {
            ext_ptrs.push(b"VK_EXT_external_memory_host\0".as_ptr().cast::<c_char>());
        }
        let mm_head: *mut c_void = if has_mm {
            (&mm_feat as *const PhysDeviceVulkanMemoryModelFeatures as *mut c_void)
        } else if has_exec {
            (&feat as *const PhysDevicePipelineExecutableFeatures as *mut c_void)
        } else {
            std::ptr::null_mut()
        };
        // f16 KV cache — 16-bit storage buffers (halves the KV footprint + attention traffic)
        let st16 = PhysDevice16BitStorageFeatures {
            s_type: ST_FEATURES_16BIT_STORAGE,
            p_next: mm_head,
            storage_buffer_16bit_access: 1,
            uniform_and_storage_buffer_16bit_access: 0,
            storage_push_constant16: 0,
            storage_input_output16: 0,
        };
        let chain: *const c_void = if has_16bit {
            (&st16 as *const PhysDevice16BitStorageFeatures).cast::<c_void>()
        } else {
            mm_head.cast_const()
        };
        // cooperative matrix — confirmed present on this device (RTX 5080: cooperativeMatrix=true,
        // 15 real shapes, GPU_PRIMITIVES.md); requested when the extension is present.
        let coopmat_feat = PhysDeviceCooperativeMatrixFeatures {
            s_type: ST_FEATURES_COOPERATIVE_MATRIX,
            p_next: chain as *mut c_void,
            cooperative_matrix: 1,
            cooperative_matrix_robust_buffer_access: 0,
        };
        let chain: *const c_void = if has_coopmat {
            (&coopmat_feat as *const PhysDeviceCooperativeMatrixFeatures).cast::<c_void>()
        } else {
            chain
        };
        // real 8-bit SSBOs (Route B for cooperative-matrix A/B — Route A's packed-uint32 pointer
        // measured wrong on the real driver, see check_coopmat/the plan's Stage-1 fork) +
        // shaderInt8 (gates SPIR-V's Int8 capability, needed regardless of storage route).
        let st8 = PhysDevice8BitStorageFeatures {
            s_type: ST_FEATURES_8BIT_STORAGE,
            p_next: chain as *mut c_void,
            storage_buffer_8bit_access: 1,
            uniform_and_storage_buffer_8bit_access: 0,
            storage_push_constant8: 0,
        };
        let chain: *const c_void = if has_8bit {
            (&st8 as *const PhysDevice8BitStorageFeatures).cast::<c_void>()
        } else {
            chain
        };
        let i8_feat = PhysDeviceShaderFloat16Int8Features {
            s_type: ST_FEATURES_SHADER_FLOAT16_INT8,
            p_next: chain as *mut c_void,
            shader_float16: 1, // the coopmat GEMM declares OpCapability Float16 (f16 A/B operands)
            shader_int8: 1,
        };
        let chain: *const c_void = if has_shaderint8 {
            (&i8_feat as *const PhysDeviceShaderFloat16Int8Features).cast::<c_void>()
        } else {
            chain
        };
        // synchronization2 feature — required to use vkCmdWriteTimestamp2 (the timestamp path);
        // prepended to the head of the feature chain
        let sync2_feat = PhysDeviceSynchronization2Features {
            s_type: ST_FEATURES_SYNCHRONIZATION2,
            p_next: chain as *mut c_void,
            synchronization2: 1,
        };
        let chain: *const c_void = if has_sync2 {
            (&sync2_feat as *const PhysDeviceSynchronization2Features).cast::<c_void>()
        } else {
            chain
        };
        // host query reset — reset the timestamp pool from the CPU (no in-cb reset ⇒ no reset/write
        // ordering hazard); required feature for vkResetQueryPool
        let hqr_feat = PhysDeviceHostQueryResetFeatures {
            s_type: ST_FEATURES_HOST_QUERY_RESET,
            p_next: chain as *mut c_void,
            host_query_reset: 1,
        };
        let chain: *const c_void = if has_hqr {
            (&hqr_feat as *const PhysDeviceHostQueryResetFeatures).cast::<c_void>()
        } else {
            chain
        };
        // ── buffer device address: the descriptor's replacement ────────────────────────────────
        // `bufferDeviceAddress` alone is not enough — dereferencing the address in SPIR-V needs
        // `OpConvertUToPtr` on a 64-bit integer, so the CORE feature `shaderInt64` comes with it.
        // Capture/replay and multi-device stay off (they are capture-tool features and cost the
        // driver an address reservation).
        let bda_feat = PhysDeviceBufferDeviceAddressFeatures {
            s_type: ST_FEATURES_BUFFER_DEVICE_ADDRESS,
            p_next: chain as *mut c_void,
            buffer_device_address: 1,
            buffer_device_address_capture_replay: 0,
            buffer_device_address_multi_device: 0,
        };
        let chain: *const c_void = if has_bda {
            (&bda_feat as *const PhysDeviceBufferDeviceAddressFeatures).cast::<c_void>()
        } else {
            chain
        };
        let mut supported = PhysicalDeviceFeatures::default();
        dev_feats_f(pd, &mut supported);
        // `shaderFloat64` gates SPIR-V's `Float64` capability: the device-resident limb route's
        // elementwise maps run in binary64.
        let has_f64 = has_bda && supported.bits[FEATURE_SHADER_FLOAT64] != 0;
        let mut core = PhysicalDeviceFeatures::default();
        if has_bda {
            core.bits[FEATURE_SHADER_INT64] = 1;
        }
        if has_f64 {
            core.bits[FEATURE_SHADER_FLOAT64] = 1;
        }
        // stated rather than assumed: the range check the SASS shows is NOT this, and it must not
        // become this by accident — the mandala mask is the only bound the kernels rely on.
        debug_assert_eq!(core.bits[FEATURE_ROBUST_BUFFER_ACCESS], 0);
        let feat2 = PhysDeviceFeatures2 {
            s_type: ST_FEATURES_2,
            p_next: chain as *mut c_void,
            features: core,
        };
        let chain: *const c_void = if has_bda {
            (&feat2 as *const PhysDeviceFeatures2).cast::<c_void>()
        } else {
            chain
        };
        let dci = DeviceCreateInfo {
            s_type: ST_DEVICE_CREATE_INFO,
            p_next: chain,
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &qci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: std::ptr::null(),
            enabled_extension_count: ext_ptrs.len() as u32,
            pp_enabled_extension_names: if ext_ptrs.is_empty() {
                std::ptr::null()
            } else {
                ext_ptrs.as_ptr()
            },
            p_enabled_features: std::ptr::null(),
        };
        let mut device: Device = std::ptr::null_mut();
        check(
            create_device(pd, &dci, std::ptr::null(), &mut device),
            "vkCreateDevice",
        );

        let get_queue = vk_fn!(gipa, instance, "vkGetDeviceQueue", GetDeviceQueue);
        let mut queue: Queue = std::ptr::null_mut();
        get_queue(device, family, 0, &mut queue);

        // timestamp entry points via vkGetDeviceProcAddr — the direct driver functions, not the
        // loader's instance trampoline (the instance-level vkCmdWriteTimestamp2 trampoline flips
        // query availability but never captures the counter on this driver → reads back 0)
        let gdpa: GetDeviceProcAddr =
            std::mem::transmute(load_fn(gipa, instance, "vkGetDeviceProcAddr"));
        let dpa = |name: &str| -> *const c_void {
            for cand in [name.to_string(), format!("{name}KHR"), format!("{name}EXT")] {
                let c = std::ffi::CString::new(cand).unwrap();
                let p = gdpa(device, c.as_ptr());
                if !p.is_null() {
                    return p;
                }
            }
            panic!("{name} (or KHR/EXT alias) not found via vkGetDeviceProcAddr");
        };
        let create_query_pool: CreateQueryPool = std::mem::transmute(dpa("vkCreateQueryPool"));
        let cmd_reset_query_pool: CmdResetQueryPool =
            std::mem::transmute(dpa("vkCmdResetQueryPool"));
        let cmd_write_timestamp2_f: CmdWriteTimestamp2 =
            std::mem::transmute(dpa("vkCmdWriteTimestamp2"));
        let cmd_pipeline_barrier2_f: CmdPipelineBarrier2 =
            std::mem::transmute(dpa("vkCmdPipelineBarrier2"));
        let get_query_results: GetQueryPoolResults =
            std::mem::transmute(dpa("vkGetQueryPoolResults"));
        let reset_query_pool_host: Option<ResetQueryPool> =
            has_hqr.then(|| std::mem::transmute::<_, ResetQueryPool>(dpa("vkResetQueryPool")));

        // a persistent pipeline cache; the driver stores its compiled cubin in the serialized blob
        let create_cache = vk_fn!(gipa, instance, "vkCreatePipelineCache", CreatePipelineCache);
        let pcci = PipelineCacheCreateInfo {
            s_type: ST_PIPELINE_CACHE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            initial_data_size: 0,
            p_initial_data: std::ptr::null(),
        };
        let mut cache: Handle = 0;
        check(
            create_cache(device, &pcci, std::ptr::null(), &mut cache),
            "vkCreatePipelineCache",
        );

        let mut mem = PhysicalDeviceMemoryProperties {
            memory_type_count: 0,
            memory_types: [MemoryType::default(); 32],
            memory_heap_count: 0,
            memory_heaps: [MemoryHeap { size: 0, flags: 0 }; 16],
        };
        mem_props_f(pd, &mut mem);

        Ok(Vk {
            instance,
            pd,
            device,
            queue,
            family,
            device_name,
            max_shared,
            mem,
            create_buffer: vk_fn!(gipa, instance, "vkCreateBuffer", CreateBuffer),
            buf_req: vk_fn!(
                gipa,
                instance,
                "vkGetBufferMemoryRequirements",
                GetBufferMemReq
            ),
            alloc_mem: vk_fn!(gipa, instance, "vkAllocateMemory", AllocateMemory),
            bind_buf: vk_fn!(gipa, instance, "vkBindBufferMemory", BindBufferMemory),
            map_mem: vk_fn!(gipa, instance, "vkMapMemory", MapMemory),
            create_shader: vk_fn!(gipa, instance, "vkCreateShaderModule", CreateShaderModule),
            create_dsl: vk_fn!(
                gipa,
                instance,
                "vkCreateDescriptorSetLayout",
                CreateDescriptorSetLayout
            ),
            create_pl: vk_fn!(
                gipa,
                instance,
                "vkCreatePipelineLayout",
                CreatePipelineLayout
            ),
            create_pipes: vk_fn!(
                gipa,
                instance,
                "vkCreateComputePipelines",
                CreateComputePipelines
            ),
            create_dpool: vk_fn!(
                gipa,
                instance,
                "vkCreateDescriptorPool",
                CreateDescriptorPool
            ),
            alloc_dsets: vk_fn!(
                gipa,
                instance,
                "vkAllocateDescriptorSets",
                AllocateDescriptorSets
            ),
            update_dsets: vk_fn!(
                gipa,
                instance,
                "vkUpdateDescriptorSets",
                UpdateDescriptorSets
            ),
            create_cpool: vk_fn!(gipa, instance, "vkCreateCommandPool", CreateCommandPool),
            alloc_cbufs: vk_fn!(
                gipa,
                instance,
                "vkAllocateCommandBuffers",
                AllocateCommandBuffers
            ),
            begin_cb: vk_fn!(gipa, instance, "vkBeginCommandBuffer", BeginCommandBuffer),
            bind_pipe: vk_fn!(gipa, instance, "vkCmdBindPipeline", CmdBindPipeline),
            bind_sets: vk_fn!(
                gipa,
                instance,
                "vkCmdBindDescriptorSets",
                CmdBindDescriptorSets
            ),
            dispatch: vk_fn!(gipa, instance, "vkCmdDispatch", CmdDispatch),
            copy_buffer: vk_fn!(gipa, instance, "vkCmdCopyBuffer", CmdCopyBuffer),
            fill_buffer: vk_fn!(gipa, instance, "vkCmdFillBuffer", CmdFillBuffer),
            pipeline_barrier: vk_fn!(gipa, instance, "vkCmdPipelineBarrier", CmdPipelineBarrier),
            end_cb: vk_fn!(gipa, instance, "vkEndCommandBuffer", EndCommandBuffer),
            reset_cb: vk_fn!(gipa, instance, "vkResetCommandBuffer", ResetCommandBuffer),
            submit: vk_fn!(gipa, instance, "vkQueueSubmit", QueueSubmit),
            wait_idle: vk_fn!(gipa, instance, "vkQueueWaitIdle", QueueWaitIdle),
            create_fence: vk_fn!(gipa, instance, "vkCreateFence", CreateFence),
            wait_fences: vk_fn!(gipa, instance, "vkWaitForFences", WaitForFences),
            reset_fences: vk_fn!(gipa, instance, "vkResetFences", ResetFences),
            destroy_fence: vk_fn!(gipa, instance, "vkDestroyFence", DestroyFence),
            timestamp_period,
            timestamp_valid_bits,
            create_query_pool,
            cmd_reset_query_pool,
            reset_query_pool_host,
            cmd_write_timestamp2_f,
            cmd_pipeline_barrier2_f,
            get_query_results,
            has_exec,
            get_exec_props: has_exec.then(|| {
                vk_fn!(
                    gipa,
                    instance,
                    "vkGetPipelineExecutablePropertiesKHR",
                    GetExecProps
                )
            }),
            get_exec_ir: has_exec.then(|| {
                vk_fn!(
                    gipa,
                    instance,
                    "vkGetPipelineExecutableInternalRepresentationsKHR",
                    GetExecIR
                )
            }),
            get_exec_stats: has_exec.then(|| {
                vk_fn!(
                    gipa,
                    instance,
                    "vkGetPipelineExecutableStatisticsKHR",
                    GetExecStats
                )
            }),
            cache,
            get_cache_data: vk_fn!(
                gipa,
                instance,
                "vkGetPipelineCacheData",
                GetPipelineCacheData
            ),
            create_cache,
            has_bda,
            get_bda: has_bda.then(|| {
                vk_fn!(
                    gipa,
                    instance,
                    "vkGetBufferDeviceAddressKHR",
                    GetBufferDeviceAddress
                )
            }),
            has_exthost,
            get_hostptr_props: has_exthost.then(|| {
                vk_fn!(
                    gipa,
                    instance,
                    "vkGetMemoryHostPointerPropertiesEXT",
                    GetMemoryHostPointerProperties
                )
            }),
            has_coopmat,
            has_f64,
        })
    }

    /// **One kernel's compiled cubin, alone.** Compile `code` against a FRESH pipeline cache and hand
    /// back that cache's serialized blob, which therefore holds exactly ONE driver container
    /// (`NVDANVVM`/`NVuc`, zstd-framed) — the SM120 SASS of this shader and nothing else.
    ///
    /// The shared [`Vk::cache`] accumulates every pipeline the process built, so its blob cannot be
    /// attributed kernel-by-kernel; this is the attribution. The temporary cache is intentionally
    /// leaked (a probe path, one per kernel, process-lifetime).
    pub unsafe fn isolated_cache_blob(&self, code: &[u8], layout: Handle) -> (Handle, Vec<u8>) {
        let pcci = PipelineCacheCreateInfo {
            s_type: ST_PIPELINE_CACHE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            initial_data_size: 0,
            p_initial_data: std::ptr::null(),
        };
        let mut cache: Handle = 0;
        check(
            (self.create_cache)(self.device, &pcci, std::ptr::null(), &mut cache),
            "vkCreatePipelineCache",
        );
        let pipe = self.pipeline_in(code, layout, cache);
        let mut size = 0usize;
        (self.get_cache_data)(self.device, cache, &mut size, std::ptr::null_mut());
        let mut buf = vec![0u8; size];
        (self.get_cache_data)(self.device, cache, &mut size, buf.as_mut_ptr().cast());
        (pipe, buf)
    }

    /// The serialized pipeline-cache blob. On NVIDIA this embeds the compiled cubin(s) — carve the
    /// ELF magic out and disassemble with `cuobjdump -sass` for real SASS, or decode directly.
    pub unsafe fn cache_data(&self) -> Vec<u8> {
        let mut size = 0usize;
        (self.get_cache_data)(self.device, self.cache, &mut size, std::ptr::null_mut());
        let mut buf = vec![0u8; size];
        (self.get_cache_data)(self.device, self.cache, &mut size, buf.as_mut_ptr().cast());
        buf
    }

    /// The driver's internal representation of a pipeline's executables — on NVIDIA, the SASS-shaped
    /// compiled form (text if `isText`, else raw bytes to disassemble). `Vec<(name, is_text, data)>`.
    /// Requires the pipeline to have been created with `PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS`.
    pub unsafe fn disasm(&self, pipeline: Handle) -> Vec<(String, bool, Vec<u8>)> {
        let Some(get_ir) = self.get_exec_ir else {
            return Vec::new();
        };
        let get_props = self.get_exec_props.unwrap();
        let pinfo = PipelineInfoKHR {
            s_type: ST_PIPELINE_INFO,
            p_next: std::ptr::null(),
            pipeline,
        };
        let mut ecount = 0u32;
        get_props(self.device, &pinfo, &mut ecount, std::ptr::null_mut());
        let mut out = Vec::new();
        for exe in 0..ecount {
            let einfo = PipelineExecutableInfoKHR {
                s_type: ST_PIPELINE_EXECUTABLE_INFO,
                p_next: std::ptr::null(),
                pipeline,
                executable_index: exe,
            };
            // MEASURED on NVIDIA 5080 / SM120: this returns **VkResult 0 (VK_SUCCESS) with count 0**
            // for every executable of every compute pipeline we build, with the extension enabled,
            // its feature bit set, and `PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS` on the
            // pipeline — the driver exposes STATISTICS ([`Vk::stats`]) but no internal
            // representation. `VK_IR_DIAG=1` prints the exact API result. The compiled SASS is NOT
            // out of reach because of that: it is in the pipeline cache
            // ([`Vk::isolated_cache_blob`] + `scripts/nvuc_sass.py`).
            let mut ir_count = 0u32;
            let rc = get_ir(self.device, &einfo, &mut ir_count, std::ptr::null_mut());
            if std::env::var_os("VK_IR_DIAG").is_some() {
                eprintln!(
                    "[VK_IR_DIAG] executables={ecount} exe={exe} \
                     vkGetPipelineExecutableInternalRepresentationsKHR -> VkResult {rc}, count {ir_count}"
                );
            }
            if ir_count == 0 {
                continue;
            }
            let blank = || PipelineExecutableInternalRepresentationKHR {
                s_type: ST_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION,
                p_next: std::ptr::null_mut(),
                name: [0; DESC_SIZE],
                description: [0; DESC_SIZE],
                is_text: 0,
                data_size: 0,
                p_data: std::ptr::null_mut(),
            };
            // pass 1: pData=null → the driver fills dataSize
            let mut irs: Vec<_> = (0..ir_count).map(|_| blank()).collect();
            get_ir(self.device, &einfo, &mut ir_count, irs.as_mut_ptr());
            // pass 2: allocate and point pData at our buffers
            let mut bufs: Vec<Vec<u8>> = irs.iter().map(|ir| vec![0u8; ir.data_size]).collect();
            for (ir, buf) in irs.iter_mut().zip(bufs.iter_mut()) {
                ir.p_data = buf.as_mut_ptr().cast();
            }
            get_ir(self.device, &einfo, &mut ir_count, irs.as_mut_ptr());
            for (ir, buf) in irs.iter().zip(bufs.into_iter()) {
                out.push((cstr(&ir.name), ir.is_text != 0, buf));
            }
        }
        out
    }

    /// Per-executable driver statistics — register count, occupancy, spills, the numbers that decide
    /// whether the kernel is issue- or bandwidth-bound. Requires `PIPELINE_CREATE_CAPTURE_STATISTICS`.
    pub unsafe fn stats(&self, pipeline: Handle) -> Vec<Stat> {
        let Some(get_stats) = self.get_exec_stats else {
            return Vec::new();
        };
        let get_props = self.get_exec_props.unwrap();
        let pinfo = PipelineInfoKHR {
            s_type: ST_PIPELINE_INFO,
            p_next: std::ptr::null(),
            pipeline,
        };
        let mut ecount = 0u32;
        get_props(self.device, &pinfo, &mut ecount, std::ptr::null_mut());
        let mut out = Vec::new();
        for exe in 0..ecount {
            let einfo = PipelineExecutableInfoKHR {
                s_type: ST_PIPELINE_EXECUTABLE_INFO,
                p_next: std::ptr::null(),
                pipeline,
                executable_index: exe,
            };
            let mut sc = 0u32;
            get_stats(self.device, &einfo, &mut sc, std::ptr::null_mut());
            let mut stats: Vec<_> = (0..sc)
                .map(|_| PipelineExecutableStatisticKHR {
                    s_type: ST_PIPELINE_EXECUTABLE_STATISTIC,
                    p_next: std::ptr::null_mut(),
                    name: [0; DESC_SIZE],
                    description: [0; DESC_SIZE],
                    format: 0,
                    value: 0,
                })
                .collect();
            get_stats(self.device, &einfo, &mut sc, stats.as_mut_ptr());
            for s in &stats {
                let value = match s.format {
                    0 => (s.value != 0).to_string(),          // BOOL32
                    1 => (s.value as i64).to_string(),        // INT64
                    2 => s.value.to_string(),                 // UINT64
                    3 => f64::from_bits(s.value).to_string(), // FLOAT64
                    _ => s.value.to_string(),
                };
                out.push(Stat {
                    name: cstr(&s.name),
                    value,
                });
            }
        }
        out
    }

    /// Create a buffer and bind fresh memory of the first type matching `props`. Returns
    /// `(buffer, memory)`; `None` if no matching memory type exists.
    pub unsafe fn buffer(&self, size: u64, usage: u32, props: u32) -> Option<(Handle, Handle)> {
        // EVERY buffer takes the device-address usage bit when the extension is live, so any buffer
        // can be handed to a `PhysicalStorageBuffer` kernel without the caller having declared its
        // intent at allocation time. The bit costs nothing on this driver (the address space is
        // assigned regardless); a per-buffer opt-in would only add a way to get it wrong.
        let usage = if self.has_bda {
            usage | BUFFER_USAGE_SHADER_DEVICE_ADDRESS
        } else {
            usage
        };
        let bci = BufferCreateInfo {
            s_type: ST_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size,
            usage,
            sharing_mode: SHARING_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };
        let mut buffer: Handle = 0;
        check(
            (self.create_buffer)(self.device, &bci, std::ptr::null(), &mut buffer),
            "vkCreateBuffer",
        );
        let mut req = MemoryRequirements::default();
        (self.buf_req)(self.device, buffer, &mut req);
        let index = (0..self.mem.memory_type_count).find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && self.mem.memory_types[i as usize].property_flags & props == props
        })?;
        // an allocation backing a SHADER_DEVICE_ADDRESS buffer must be flagged as such
        let mafi = MemoryAllocateFlagsInfo {
            s_type: ST_MEMORY_ALLOCATE_FLAGS_INFO,
            p_next: std::ptr::null(),
            flags: MEMORY_ALLOCATE_DEVICE_ADDRESS,
            device_mask: 0,
        };
        let mai = MemoryAllocateInfo {
            s_type: ST_MEMORY_ALLOCATE_INFO,
            p_next: if self.has_bda {
                (&mafi as *const MemoryAllocateFlagsInfo).cast::<c_void>()
            } else {
                std::ptr::null()
            },
            allocation_size: req.size,
            memory_type_index: index,
        };
        let mut memory: Handle = 0;
        check(
            (self.alloc_mem)(self.device, &mai, std::ptr::null(), &mut memory),
            "vkAllocateMemory",
        );
        check(
            (self.bind_buf)(self.device, buffer, memory, 0),
            "vkBindBufferMemory",
        );
        Some((buffer, memory))
    }

    /// **A buffer whose memory is an `mmap` of a FILE — the KV's disk tier.**
    ///
    /// `buffer(HOST_VISIBLE)` gets anonymous driver-owned pages: not file-backed, so the kernel
    /// cannot page them out, and a KV bigger than free RAM OOM-kills instead of spilling. This
    /// adopts a `MAP_SHARED` mapping of `path` instead, so the same device-readable buffer is
    /// PAGE-CACHE backed — cold blocks go to the SSD under pressure and come back on touch, with no
    /// swap involved and no copy in the gather path.
    ///
    /// Returns `None` when `VK_EXT_external_memory_host` is absent, when the file cannot be sized,
    /// or when the driver's `memoryTypeBits` for an imported pointer share nothing with the
    /// buffer's requirements — the last is ASKED (`vkGetMemoryHostPointerPropertiesEXT`), because
    /// choosing a type outside that mask is undefined behaviour rather than a slow path.
    ///
    /// # Safety
    /// The mapping is leaked deliberately: the imported allocation must outlive every submit that
    /// reads it, and this buffer lives for the process. `size` is rounded UP to a page, which the
    /// import alignment requires (`minImportedHostPointerAlignment`, 4 KB on this device).
    pub unsafe fn buffer_from_host_ptr(
        &self,
        path: &std::path::Path,
        size: u64,
        usage: u32,
    ) -> Option<(Handle, Handle, *mut c_void)> {
        self.buffer_from_mapping(Some(path), size, usage)
    }

    /// [`Self::buffer_from_host_ptr`] over an ANONYMOUS mapping — no file, and the question it
    /// answers is whether the pages are committed LAZILY.
    ///
    /// `vkAllocateMemory(HOST_VISIBLE)` commits eagerly: a 1M-position KV cap costs its full
    /// footprint in resident memory the moment it is allocated, whether or not the context ever
    /// reaches 1M. An anonymous `mmap` is lazy by construction — pages materialise on first touch —
    /// so IF the driver leaves an imported anonymous mapping unpinned, a big `kv_cap` becomes
    /// address space rather than resident memory and the tier only pays for the context actually
    /// used. If it pins on import, this is identical to the eager path and says so.
    ///
    /// # Safety
    /// as [`Self::buffer_from_host_ptr`].
    pub unsafe fn buffer_from_anon_ptr(
        &self,
        size: u64,
        usage: u32,
    ) -> Option<(Handle, Handle, *mut c_void)> {
        self.buffer_from_mapping(None, size, usage)
    }

    /// # Safety
    /// as [`Self::buffer_from_host_ptr`].
    unsafe fn buffer_from_mapping(
        &self,
        path: Option<&std::path::Path>,
        size: u64,
        usage: u32,
    ) -> Option<(Handle, Handle, *mut c_void)> {
        // Each refusal names ITSELF. A single `None` for five different causes is how an hour goes
        // into re-running a twelve-minute load to find out which one it was.
        if !self.has_exthost {
            eprintln!("host-ptr: VK_EXT_external_memory_host not present/enabled");
            return None;
        }
        let Some(props_fn) = self.get_hostptr_props else {
            eprintln!("host-ptr: vkGetMemoryHostPointerPropertiesEXT did not load");
            return None;
        };
        const PAGE: u64 = 4096;
        let bytes = size.div_ceil(PAGE) * PAGE;
        // a sparse file: the holes cost nothing until a block is actually written
        let file = match path {
            None => None,
            Some(path) => {
                let f = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path);
                let f = match f {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("host-ptr: open {}: {e}", path.display());
                        return None;
                    }
                };
                if let Err(e) = f.set_len(bytes) {
                    eprintln!("host-ptr: set_len {bytes}: {e}");
                    return None;
                }
                Some(f)
            }
        };
        let fd = file
            .as_ref()
            .map_or(-1, std::os::unix::io::AsRawFd::as_raw_fd);
        let ptr = mmap(
            std::ptr::null_mut(),
            bytes as usize,
            PROT_READ | PROT_WRITE,
            if fd < 0 { MAP_SHARED_ANON } else { MAP_SHARED },
            fd,
            0,
        );
        if ptr as isize == -1 {
            eprintln!("host-ptr: mmap {bytes} bytes failed");
            return None;
        }
        // The fd may close; the mapping keeps the file alive. Leaked with the mapping, on purpose.
        std::mem::forget(file);
        let _ = path;

        // ── NO buffer-device-address on an IMPORTED allocation ─────────────────────────────────
        // `buffer()` gives every buffer the BDA usage bit, which then REQUIRES
        // `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT` on its allocation. Combined with an imported host
        // pointer this driver refuses the allocation outright: `vkAllocateMemory(imported, type 2,
        // 2097152 B) -> -2` (OUT_OF_DEVICE_MEMORY) on a 2 MB import, measured. The KV planes are
        // bound as ordinary storage-buffer descriptors, never dereferenced through a raw address,
        // so dropping the bit costs this buffer nothing.
        let bci = BufferCreateInfo {
            s_type: ST_BUFFER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            size: bytes,
            usage,
            sharing_mode: SHARING_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: std::ptr::null(),
        };
        let mut buffer: Handle = 0;
        check(
            (self.create_buffer)(self.device, &bci, std::ptr::null(), &mut buffer),
            "vkCreateBuffer(host-ptr)",
        );
        let mut req = MemoryRequirements::default();
        (self.buf_req)(self.device, buffer, &mut req);

        // ASK the driver which memory types may back this pointer, and intersect with the buffer's.
        let mut hp = MemoryHostPointerProperties {
            s_type: ST_MEMORY_HOST_POINTER_PROPERTIES,
            p_next: std::ptr::null_mut(),
            memory_type_bits: 0,
        };
        let r = props_fn(
            self.device,
            EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION,
            ptr,
            &mut hp,
        );
        if r != 0 {
            eprintln!("host-ptr: vkGetMemoryHostPointerPropertiesEXT -> {r}");
            munmap(ptr, bytes as usize);
            return None;
        }
        let allowed = req.memory_type_bits & hp.memory_type_bits;
        let imp = ImportMemoryHostPointerInfo {
            s_type: ST_IMPORT_MEMORY_HOST_POINTER_INFO,
            p_next: std::ptr::null(),
            handle_type: EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION,
            p_host_pointer: ptr,
        };
        // ── TRY EVERY ALLOWED TYPE, do not guess one ───────────────────────────────────────────
        // This picked the first HOST_VISIBLE type in the mask and failed with
        // `VK_ERROR_OUT_OF_DEVICE_MEMORY` on a 2 MB import, which reads exactly like "the driver
        // refuses this". It did not: on this device the mask is 0xc, type 2
        // (HOST_VISIBLE|HOST_COHERENT) is REFUSED and type 3 (…|HOST_CACHED) succeeds — importing
        // ordinary host pages needs the CACHED type, since that is what they are. A single guess
        // cannot tell "wrong type" from "unsupported", so this reports every attempt.
        let mut memory: Handle = 0;
        let mut chosen = u32::MAX;
        for i in 0..self.mem.memory_type_count {
            if allowed & (1 << i) == 0 {
                continue;
            }
            let mai = MemoryAllocateInfo {
                s_type: ST_MEMORY_ALLOCATE_INFO,
                p_next: (&imp as *const ImportMemoryHostPointerInfo).cast::<c_void>(),
                // the imported size must be the MAPPING's, not the buffer's requirement
                allocation_size: bytes,
                memory_type_index: i,
            };
            if (self.alloc_mem)(self.device, &mai, std::ptr::null(), &mut memory) == 0 {
                chosen = i;
                break;
            }
        }
        if chosen == u32::MAX {
            eprintln!(
                "host-ptr: no memory type accepted the import (buffer 0x{:x} & imported 0x{:x} \
= 0x{allowed:x}, {bytes} B)",
                req.memory_type_bits, hp.memory_type_bits
            );
            munmap(ptr, bytes as usize);
            return None;
        }
        check(
            (self.bind_buf)(self.device, buffer, memory, 0),
            "vkBindBufferMemory(host-ptr)",
        );
        Some((buffer, memory, ptr))
    }

    /// **The control for [`Self::buffer_from_host_ptr`]** — the identical import against an
    /// ANONYMOUS mapping, which the driver can pin. Separates "the import path is wrong" from "the
    /// driver refuses evictable memory", which is the only question that decides whether a disk
    /// tier can be a mapping at all. See `examples/host_ptr_probe.rs`.
    ///
    /// # Safety
    /// Allocates and immediately leaks a small probe mapping; call once, at startup, for diagnosis.
    pub unsafe fn import_anon_probe(&self, bytes: u64) -> bool {
        if !self.has_exthost {
            return false;
        }
        let Some(props_fn) = self.get_hostptr_props else {
            return false;
        };
        let ptr = mmap(
            std::ptr::null_mut(),
            bytes as usize,
            PROT_READ | PROT_WRITE,
            MAP_SHARED_ANON,
            -1,
            0,
        );
        if ptr as isize == -1 {
            eprintln!("anon-probe: mmap failed");
            return false;
        }
        let mut hp = MemoryHostPointerProperties {
            s_type: ST_MEMORY_HOST_POINTER_PROPERTIES,
            p_next: std::ptr::null_mut(),
            memory_type_bits: 0,
        };
        let r = props_fn(
            self.device,
            EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION,
            ptr,
            &mut hp,
        );
        if r != 0 {
            eprintln!("anon-probe: hostPointerProperties -> {r}");
            munmap(ptr, bytes as usize);
            return false;
        }
        eprintln!(
            "anon-probe: driver allows memory types 0x{:x} for an imported pointer; {} types total",
            hp.memory_type_bits, self.mem.memory_type_count
        );
        // Try EVERY type the driver says is valid, not the first plausible one — a single guess
        // that fails cannot distinguish "wrong type" from "import unsupported in practice".
        let imp = ImportMemoryHostPointerInfo {
            s_type: ST_IMPORT_MEMORY_HOST_POINTER_INFO,
            p_next: std::ptr::null(),
            handle_type: EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION,
            p_host_pointer: ptr,
        };
        let mut ok = false;
        for i in 0..self.mem.memory_type_count {
            if hp.memory_type_bits & (1 << i) == 0 {
                continue;
            }
            let mai = MemoryAllocateInfo {
                s_type: ST_MEMORY_ALLOCATE_INFO,
                p_next: (&imp as *const ImportMemoryHostPointerInfo).cast::<c_void>(),
                allocation_size: bytes,
                memory_type_index: i,
            };
            let mut memory: Handle = 0;
            let ar = (self.alloc_mem)(self.device, &mai, std::ptr::null(), &mut memory);
            eprintln!(
                "anon-probe:   type {i} (flags 0x{:x}) -> {ar}",
                self.mem.memory_types[i as usize].property_flags
            );
            if ar == 0 {
                ok = true;
            }
        }
        if !ok {
            munmap(ptr, bytes as usize);
        }
        ok
    }

    /// **The buffer's GPU address** — the number a `PhysicalStorageBuffer` kernel dereferences.
    /// `None` when `VK_KHR_buffer_device_address` is not live, in which case every kernel stays on
    /// the descriptor path. The address is stable for the buffer's lifetime, so it is read once at
    /// plan time and never per dispatch.
    ///
    /// **What it removes.** A descriptor carries a base AND a range, and the driver spends one
    /// `ISETP.GE.U32` per global load comparing the index against that range (measured in the
    /// `gemv_dp4a_sb` hot loop at 7.4 % of issue slots, with `robustBufferAccess` DISABLED). A raw
    /// address has no range to compare — which makes the mandala mask (`MandalaMemory.access`:
    /// `base + (key &&& (2^k − 1))`) the ONLY bound, exactly as `masked_load_never_oob` states it.
    pub unsafe fn device_address(&self, buffer: Handle) -> Option<u64> {
        let f = self.get_bda?;
        let info = BufferDeviceAddressInfo {
            s_type: ST_BUFFER_DEVICE_ADDRESS_INFO,
            p_next: std::ptr::null(),
            buffer,
        };
        Some(f(self.device, &info))
    }

    /// Map a whole allocation.
    pub unsafe fn map(&self, memory: Handle) -> *mut c_void {
        let mut p: *mut c_void = std::ptr::null_mut();
        check(
            (self.map_mem)(self.device, memory, 0, WHOLE_SIZE, 0, &mut p),
            "vkMapMemory",
        );
        p
    }

    /// One descriptor set binding `buffers[i]` at (set 0, binding i), all storage buffers.
    /// Returns `(pipeline_layout, descriptor_set)`.
    pub unsafe fn ssbo_set(&self, buffers: &[Handle]) -> (Handle, Handle) {
        let n = buffers.len() as u32;
        let bindings: Vec<DescriptorSetLayoutBinding> = (0..n)
            .map(|i| DescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: SHADER_STAGE_COMPUTE,
                p_immutable_samplers: std::ptr::null(),
            })
            .collect();
        let dslci = DescriptorSetLayoutCreateInfo {
            s_type: ST_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            binding_count: n,
            p_bindings: bindings.as_ptr(),
        };
        let mut dsl: Handle = 0;
        check(
            (self.create_dsl)(self.device, &dslci, std::ptr::null(), &mut dsl),
            "vkCreateDescriptorSetLayout",
        );
        let plci = PipelineLayoutCreateInfo {
            s_type: ST_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &dsl,
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
        };
        let mut layout: Handle = 0;
        check(
            (self.create_pl)(self.device, &plci, std::ptr::null(), &mut layout),
            "vkCreatePipelineLayout",
        );
        let dps = DescriptorPoolSize {
            ty: DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: n,
        };
        let dpci = DescriptorPoolCreateInfo {
            s_type: ST_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &dps,
        };
        let mut dpool: Handle = 0;
        check(
            (self.create_dpool)(self.device, &dpci, std::ptr::null(), &mut dpool),
            "vkCreateDescriptorPool",
        );
        let dsai = DescriptorSetAllocateInfo {
            s_type: ST_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: dpool,
            descriptor_set_count: 1,
            p_set_layouts: &dsl,
        };
        let mut dset: Handle = 0;
        check(
            (self.alloc_dsets)(self.device, &dsai, &mut dset),
            "vkAllocateDescriptorSets",
        );
        let infos: Vec<DescriptorBufferInfo> = buffers
            .iter()
            .map(|&b| DescriptorBufferInfo {
                buffer: b,
                offset: 0,
                range: WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<WriteDescriptorSet> = (0..n)
            .map(|i| WriteDescriptorSet {
                s_type: ST_WRITE_DESCRIPTOR_SET,
                p_next: std::ptr::null(),
                dst_set: dset,
                dst_binding: i,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: DESCRIPTOR_TYPE_STORAGE_BUFFER,
                p_image_info: std::ptr::null(),
                p_buffer_info: &infos[i as usize],
                p_texel_buffer_view: std::ptr::null(),
            })
            .collect();
        (self.update_dsets)(self.device, n, writes.as_ptr(), 0, std::ptr::null());
        (layout, dset)
    }

    /// A compute pipeline from SPIR-V bytes over `layout`, entry `main`.
    pub unsafe fn pipeline(&self, code: &[u8], layout: Handle) -> Handle {
        self.pipeline_in(code, layout, self.cache)
    }

    /// [`Vk::pipeline`] against a NAMED pipeline cache — the only difference is where the driver
    /// files the compiled cubin, which is what [`Vk::isolated_cache_blob`] needs.
    pub unsafe fn pipeline_in(&self, code: &[u8], layout: Handle, cache: Handle) -> Handle {
        let smci = ShaderModuleCreateInfo {
            s_type: ST_SHADER_MODULE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            code_size: code.len(),
            p_code: code.as_ptr().cast::<u32>(),
        };
        let mut module: Handle = 0;
        check(
            (self.create_shader)(self.device, &smci, std::ptr::null(), &mut module),
            "vkCreateShaderModule",
        );
        let cpci = ComputePipelineCreateInfo {
            s_type: ST_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            // capture the SASS/IR + stats when the extension is live (free — it only preserves the
            // driver's compiled form for later readout, no effect on the pipeline that runs)
            flags: if self.has_exec {
                PIPELINE_CREATE_CAPTURE_STATISTICS
                    | PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS
            } else {
                0
            },
            stage: PipelineShaderStageCreateInfo {
                s_type: ST_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: std::ptr::null(),
                flags: 0,
                stage: SHADER_STAGE_COMPUTE,
                module,
                p_name: b"main\0".as_ptr().cast::<c_char>(),
                p_specialization_info: std::ptr::null(),
            },
            layout,
            base_pipeline_handle: 0,
            base_pipeline_index: -1,
        };
        let mut pipeline: Handle = 0;
        check(
            (self.create_pipes)(
                self.device,
                cache,
                1,
                &cpci,
                std::ptr::null(),
                &mut pipeline,
            ),
            "vkCreateComputePipelines",
        );
        pipeline
    }

    /// Allocate one primary command buffer from a pool created with the RESET_COMMAND_BUFFER flag
    /// (bit 1, vk.xml) so the single buffer can be re-recorded every submit — the persistent-command
    /// buffer that kills the per-dispatch `vkCreateCommandPool` (MILESTONE 3: a training step fires
    /// dozens of dispatches; a fresh pool each was the bench's dominant host cost). Reset with
    /// [`Vk::reset_cb`] before each re-begin.
    pub unsafe fn reusable_command_buffer(&self) -> CommandBuffer {
        let cpci = CommandPoolCreateInfo {
            s_type: ST_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0x2, // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
            queue_family_index: self.family,
        };
        let mut cpool: Handle = 0;
        check(
            (self.create_cpool)(self.device, &cpci, std::ptr::null(), &mut cpool),
            "vkCreateCommandPool",
        );
        let cbai = CommandBufferAllocateInfo {
            s_type: ST_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: cpool,
            level: COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cb: CommandBuffer = std::ptr::null_mut();
        check(
            (self.alloc_cbufs)(self.device, &cbai, &mut cb),
            "vkAllocateCommandBuffers",
        );
        cb
    }

    /// Reset a command buffer for re-recording (the pool must allow individual resets).
    pub unsafe fn reset_cb(&self, cb: CommandBuffer) {
        check((self.reset_cb)(cb, 0), "vkResetCommandBuffer");
    }

    /// Allocate one primary command buffer from a fresh pool.
    pub unsafe fn command_buffer(&self) -> CommandBuffer {
        let cpci = CommandPoolCreateInfo {
            s_type: ST_COMMAND_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            queue_family_index: self.family,
        };
        let mut cpool: Handle = 0;
        check(
            (self.create_cpool)(self.device, &cpci, std::ptr::null(), &mut cpool),
            "vkCreateCommandPool",
        );
        let cbai = CommandBufferAllocateInfo {
            s_type: ST_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: cpool,
            level: COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cb: CommandBuffer = std::ptr::null_mut();
        check(
            (self.alloc_cbufs)(self.device, &cbai, &mut cb),
            "vkAllocateCommandBuffers",
        );
        cb
    }

    pub unsafe fn begin(&self, cb: CommandBuffer) {
        let begin = CommandBufferBeginInfo {
            s_type: ST_COMMAND_BUFFER_BEGIN_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            p_inheritance_info: std::ptr::null(),
        };
        check((self.begin_cb)(cb, &begin), "vkBeginCommandBuffer");
    }

    /// A global memory barrier between pipeline stages.
    pub unsafe fn barrier(
        &self,
        cb: CommandBuffer,
        src_stage: u32,
        dst_stage: u32,
        src_access: u32,
        dst_access: u32,
    ) {
        let mb = MemoryBarrier {
            s_type: ST_MEMORY_BARRIER,
            p_next: std::ptr::null(),
            src_access_mask: src_access,
            dst_access_mask: dst_access,
        };
        (self.pipeline_barrier)(
            cb,
            src_stage,
            dst_stage,
            0,
            1,
            &mb,
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
        );
    }

    /// A global memory barrier via synchronization2 (vkCmdPipelineBarrier2) — the stage/access words
    /// are 64-bit and live inside the barrier. Different driver code path from the legacy `barrier`;
    /// the question this measures is whether NVIDIA compiles it to a cheaper inter-dispatch sync.
    pub unsafe fn barrier2(
        &self,
        cb: CommandBuffer,
        src_stage: u64,
        dst_stage: u64,
        src_access: u64,
        dst_access: u64,
    ) {
        let mb = MemoryBarrier2 {
            s_type: ST_MEMORY_BARRIER_2,
            p_next: std::ptr::null(),
            src_stage_mask: src_stage,
            src_access_mask: src_access,
            dst_stage_mask: dst_stage,
            dst_access_mask: dst_access,
        };
        let dep = DependencyInfo {
            s_type: ST_DEPENDENCY_INFO,
            p_next: std::ptr::null(),
            dependency_flags: 0,
            memory_barrier_count: 1,
            p_memory_barriers: &mb,
            buffer_memory_barrier_count: 0,
            p_buffer_memory_barriers: std::ptr::null(),
            image_memory_barrier_count: 0,
            p_image_memory_barriers: std::ptr::null(),
        };
        (self.cmd_pipeline_barrier2_f)(cb, &dep);
    }

    /// Submit one command buffer and wait for the queue to drain.
    pub unsafe fn submit_wait(&self, cb: CommandBuffer) {
        let si = SubmitInfo {
            s_type: ST_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cb,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        check((self.submit)(self.queue, 1, &si, 0), "vkQueueSubmit");
        check((self.wait_idle)(self.queue), "vkQueueWaitIdle");
    }

    /// A fresh timestamp query pool holding `count` queries (VK_QUERY_TYPE_TIMESTAMP).
    pub unsafe fn timestamp_pool(&self, count: u32) -> QueryPool {
        let qpci = QueryPoolCreateInfo {
            s_type: ST_QUERY_POOL_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            query_type: QUERY_TYPE_TIMESTAMP,
            query_count: count,
            pipeline_statistics: 0,
        };
        let mut handle: Handle = 0;
        check(
            (self.create_query_pool)(self.device, &qpci, std::ptr::null(), &mut handle),
            "vkCreateQueryPool",
        );
        QueryPool { handle, count }
    }

    /// Reset every query in `pool` (recorded into `cb`; must precede the writes each submit).
    pub unsafe fn cmd_reset_timestamps(&self, cb: CommandBuffer, pool: &QueryPool) {
        (self.cmd_reset_query_pool)(cb, pool.handle, 0, pool.count);
    }

    /// Host-side reset of `pool` (no command buffer). `true` if host query reset is available.
    pub unsafe fn reset_timestamps_host(&self, pool: &QueryPool) -> bool {
        match self.reset_query_pool_host {
            Some(f) => {
                f(self.device, pool.handle, 0, pool.count);
                true
            }
            None => false,
        }
    }

    /// Write a timestamp latched at `stage` (TOP/BOTTOM_OF_PIPE, a sync2 u64 flag) into
    /// `pool[query]` (recorded). Uses vkCmdWriteTimestamp2 (the legacy call is stubbed here).
    pub unsafe fn cmd_write_timestamp(
        &self,
        cb: CommandBuffer,
        pool: &QueryPool,
        stage: u64,
        query: u32,
    ) {
        (self.cmd_write_timestamp2_f)(cb, stage, pool.handle, query);
    }

    /// Read all `count` timestamps back as u64 ticks, waiting for them to be available.
    /// GPU-active µs between two ticks = `(b - a) as f64 * timestamp_period / 1000.0`.
    pub unsafe fn read_timestamps(&self, pool: &QueryPool) -> Vec<u64> {
        self.read_timestamps_prefix(pool, pool.count)
    }

    /// Read the first `queries` timestamps of `pool` back as u64 ticks, waiting for them to be
    /// available. A stamped tape fills a prefix of its pool, so a caller reads exactly the prefix it
    /// wrote: `QUERY_RESULT_WAIT` blocks until every query asked for holds a value, and a query that
    /// was never written never does.
    pub unsafe fn read_timestamps_prefix(&self, pool: &QueryPool, queries: u32) -> Vec<u64> {
        let queries = queries.min(pool.count);
        let mut out = vec![0u64; queries as usize];
        (self.get_query_results)(
            self.device,
            pool.handle,
            0,
            queries,
            queries as usize * 8,
            out.as_mut_ptr().cast(),
            8,
            QUERY_RESULT_64 | QUERY_RESULT_WAIT,
        );
        out
    }

    /// Submit `cb`, signalling a fresh fence, and wait for it up to `timeout_ns`. Returns `true`
    /// when the fence signalled (the kernel finished), `false` on VK_TIMEOUT (a hung/deadlocked
    /// kernel — never an infinite wait). On timeout the queue is still executing the hung work, so
    /// the caller must NOT reuse this device (tear the process down instead). The fence is destroyed
    /// only on the signalled path (destroying a pending fence is illegal).
    pub unsafe fn submit_timeout(&self, cb: CommandBuffer, timeout_ns: u64) -> bool {
        let fci = FenceCreateInfo {
            s_type: ST_FENCE_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: 0,
        };
        let mut fence: Handle = 0;
        check(
            (self.create_fence)(self.device, &fci, std::ptr::null(), &mut fence),
            "vkCreateFence",
        );
        let si = SubmitInfo {
            s_type: ST_SUBMIT_INFO,
            p_next: std::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: std::ptr::null(),
            p_wait_dst_stage_mask: std::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cb,
            signal_semaphore_count: 0,
            p_signal_semaphores: std::ptr::null(),
        };
        check((self.submit)(self.queue, 1, &si, fence), "vkQueueSubmit");
        let r = (self.wait_fences)(self.device, 1, &fence, 1, timeout_ns);
        if r == VK_SUCCESS {
            (self.destroy_fence)(self.device, fence, std::ptr::null());
            true
        } else {
            // VK_TIMEOUT (or error): the kernel is still in flight — leak the fence, signal deadlock.
            false
        }
    }
}
