//! Writable file-backed storage for the key/value planes.
//!
//! A [`KvState`](super::KvState) plane is normally an owned `Vec<T>`. [`KvPlane`] adds a second
//! backing: a writable `MAP_SHARED` mapping of a fully allocated cache file. The words the
//! attention path reads and writes are the same words; only the memory they live in changes. The
//! mapping is file-backed, so the kernel can write a clean page back and evict it under memory
//! pressure instead of OOM-killing the process, and the cache can be larger than RAM.
//!
//! # Domain
//!
//! - The file is a cache. Its bytes are host-endian Q32.32 or binary32 words and carry no header;
//!   the geometry is the caller's, and [`KvMap::create`] truncates the file before mapping it.
//! - The file is allocated to its full extent with `posix_fallocate` at creation. A capacity the
//!   filesystem cannot hold is refused there; it is never discovered as `SIGBUS` on a later store.
//! - The file must not be on a tmpfs, which holds every page in RAM and defeats the point.
//! - `clone` of a mapped plane copies it into owned memory. Two states never share a writable
//!   mapping, so a cloned state cannot overwrite the file the original reads.
//! - Linux only. Other targets have no mapped variant, and [`KvMap::create`] refuses.

use core::marker::PhantomData;
use core::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A refused file-backed plane.
#[derive(Debug)]
pub enum KvFileError {
    /// The cache file could not be opened, allocated or mapped.
    Io(std::io::Error),
    /// A zero-length file or plane.
    Empty,
    /// A byte length, offset or element count left `usize`.
    Overflow,
    /// A plane range is not inside the mapping, or its offset is not aligned for the element.
    BadRange,
    /// This target has no writable mapping.
    Unsupported,
}

impl core::fmt::Display for KvFileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "kv file: {e}"),
            Self::Empty => f.write_str("kv file: zero-length file or plane"),
            Self::Overflow => f.write_str("kv file: byte length left usize"),
            Self::BadRange => f.write_str("kv file: plane range outside the mapping"),
            Self::Unsupported => f.write_str("kv file: writable mappings are Linux-only"),
        }
    }
}

impl std::error::Error for KvFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Empty | Self::Overflow | Self::BadRange | Self::Unsupported => None,
        }
    }
}

impl From<std::io::Error> for KvFileError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A writable shared mapping of one cache file, shared by every plane that points into it.
#[cfg(target_os = "linux")]
pub(crate) struct KvMap {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping is one contiguous region. `KvState` hands out `&[T]` only through `&self`
// and `&mut [T]` only through `&mut self`, so no read and write overlap; every plane range is a
// disjoint sub-range of the same mapping. `munmap` runs once, when the last `Arc` drops.
#[cfg(target_os = "linux")]
unsafe impl Send for KvMap {}
#[cfg(target_os = "linux")]
unsafe impl Sync for KvMap {}

#[cfg(target_os = "linux")]
impl KvMap {
    /// Create (or truncate) `path`, allocate `len` bytes, and map them read-write and shared.
    ///
    /// # Errors
    ///
    /// [`KvFileError::Empty`] for a zero length, [`KvFileError::Overflow`] when `len` leaves
    /// `off_t`, and [`KvFileError::Io`] when the open, `posix_fallocate` or `mmap` refuses. The
    /// allocation is the capacity check: a filesystem without room refuses here, before any token
    /// is stored.
    ///
    /// # Complexity
    ///
    /// One open, one `ftruncate`-equivalent allocation and one `mmap`: `O(1)` file operations. The
    /// allocation reserves extents, so its cost is the filesystem's, not the page count's.
    pub(crate) fn create(path: &str, len: usize) -> Result<Self, KvFileError> {
        use std::os::unix::io::AsRawFd;

        if len == 0 {
            return Err(KvFileError::Empty);
        }
        let len_off = i64::try_from(len).map_err(|_| KvFileError::Overflow)?;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        // `posix_fallocate` returns an errno rather than setting `errno` on failure. Allocating the
        // full extent is what turns a full filesystem into a refusal here instead of a SIGBUS on a
        // later store into the mapping.
        let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len_off) };
        if rc != 0 {
            return Err(KvFileError::Io(std::io::Error::from_raw_os_error(rc)));
        }
        // SAFETY: the descriptor is open for the call, `len` is the allocated extent, the mapping
        // is shared and read-write, and the returned pointer is checked against `MAP_FAILED`
        // before it is stored. The mapping outlives `file`, which may close here: `mmap` holds its
        // own reference to the file.
        let ptr = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(KvFileError::Io(std::io::Error::last_os_error()));
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    /// Bytes of the mapping.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Write every dirty page back to the file and wait for the writes.
    ///
    /// # Errors
    ///
    /// The mapping's own `msync` refusal.
    ///
    /// # Complexity
    ///
    /// `O(dirty pages)` I/O; one `msync` call.
    pub(crate) fn sync(&self) -> Result<(), KvFileError> {
        // SAFETY: the range is the live mapping and `MS_SYNC` asks for one pass of writeback.
        let rc = unsafe { libc::msync(self.ptr.cast(), self.len, libc::MS_SYNC) };
        if rc == 0 {
            Ok(())
        } else {
            Err(KvFileError::Io(std::io::Error::last_os_error()))
        }
    }
}

#[cfg(target_os = "linux")]
impl core::fmt::Debug for KvMap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KvMap")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(target_os = "linux")]
impl Drop for KvMap {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` are the live mapping made by `create`, unmapped once here.
        unsafe {
            libc::munmap(self.ptr.cast(), self.len);
        }
    }
}

/// A non-Linux `KvMap` exists only so `KvState` has one field type on every target. It is never
/// constructed: [`KvMap::create`] refuses, and [`KvPlane`] has no mapped variant.
#[cfg(not(target_os = "linux"))]
#[derive(Debug)]
pub(crate) struct KvMap {
    len: usize,
}

#[cfg(not(target_os = "linux"))]
impl KvMap {
    pub(crate) fn create(_path: &str, _len: usize) -> Result<Self, KvFileError> {
        Err(KvFileError::Unsupported)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn sync(&self) -> Result<(), KvFileError> {
        Err(KvFileError::Unsupported)
    }
}

/// One key or value plane: an owned vector or a disjoint range of a shared writable mapping.
pub struct KvPlane<T> {
    inner: Inner<T>,
}

enum Inner<T> {
    Owned(Vec<T>),
    #[cfg(target_os = "linux")]
    Mapped {
        map: Arc<KvMap>,
        /// Byte offset of the first element inside `map`.
        off: usize,
        /// Element count.
        len: usize,
        elem: PhantomData<T>,
    },
}

impl<T> KvPlane<T> {
    /// Wrap an owned plane.
    pub(crate) fn owned(words: Vec<T>) -> Self {
        Self {
            inner: Inner::Owned(words),
        }
    }

    /// Point a plane at `[off, off + len · size_of::<T>())` inside `map`.
    ///
    /// # Errors
    ///
    /// [`KvFileError::BadRange`] when the offset's alignment or the range is outside `map`.
    ///
    /// # Complexity
    ///
    /// `O(1)` checked arithmetic.
    #[cfg(target_os = "linux")]
    pub(crate) fn mapped(map: Arc<KvMap>, off: usize, len: usize) -> Result<Self, KvFileError> {
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(KvFileError::Overflow)?;
        let end = off.checked_add(bytes).ok_or(KvFileError::Overflow)?;
        if !off.is_multiple_of(core::mem::align_of::<T>()) || end > map.len() {
            return Err(KvFileError::BadRange);
        }
        Ok(Self {
            inner: Inner::Mapped {
                map,
                off,
                len,
                elem: PhantomData,
            },
        })
    }

    /// The byte length of the backing file, or `None` for an owned plane.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn mapped_bytes(&self) -> Option<usize> {
        match &self.inner {
            Inner::Owned(_) => None,
            #[cfg(target_os = "linux")]
            Inner::Mapped { map, .. } => Some(map.len()),
        }
    }
}

impl<T> Deref for KvPlane<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match &self.inner {
            Inner::Owned(words) => words,
            #[cfg(target_os = "linux")]
            Inner::Mapped { map, off, len, .. } => {
                // SAFETY: `mapped` checked the alignment and extent against the live mapping, the
                // mapping is writable but this borrow is shared, and the slice does not outlive
                // `self`.
                unsafe { core::slice::from_raw_parts(map.ptr.add(*off).cast::<T>(), *len) }
            }
        }
    }
}

impl<T> DerefMut for KvPlane<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.inner {
            Inner::Owned(words) => words,
            #[cfg(target_os = "linux")]
            Inner::Mapped { map, off, len, .. } => {
                // SAFETY: as `deref`, plus `&mut self`: this plane range is disjoint from every
                // other plane range in the same mapping, and no shared borrow of this plane exists
                // while the exclusive borrow is live.
                unsafe { core::slice::from_raw_parts_mut(map.ptr.add(*off).cast::<T>(), *len) }
            }
        }
    }
}

impl<T: Clone> Clone for KvPlane<T> {
    fn clone(&self) -> Self {
        // A mapped plane is materialised. Two writable states never share a mapping, so a cloned
        // state cannot overwrite the file the original reads. The cost is one plane copy; the
        // prefix cache is the only clone site, and it is not the serving path.
        Self::owned(self.to_vec())
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for KvPlane<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.inner {
            Inner::Owned(words) => words.fmt(f),
            #[cfg(target_os = "linux")]
            Inner::Mapped { len, .. } => write!(f, "Mapped({len} words)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!(
            "apeiron-kvplane-{tag}-{}-{nanos}.bin",
            std::process::id()
        ))
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_mapped_plane_reads_and_writes_the_file() {
        let path = temp_path("roundtrip");
        let path_s = path.to_str().expect("utf-8 temp path");
        let map = Arc::new(KvMap::create(path_s, 4 * 8).expect("mapping"));
        let mut plane = KvPlane::<i64>::mapped(Arc::clone(&map), 0, 4).expect("plane");
        assert_eq!(&plane[..], &[0, 0, 0, 0]);
        plane[1] = -7;
        plane[3] = i64::MAX;
        map.sync().expect("sync");
        let raw = std::fs::read(&path).expect("read back");
        assert_eq!(raw.len(), 32);
        let mut words = [0i64; 4];
        for (i, chunk) in raw.as_chunks::<8>().0.iter().enumerate() {
            words[i] = i64::from_le_bytes(*chunk);
        }
        assert_eq!(words, [0, -7, 0, i64::MAX]);
        drop(plane);
        drop(map);
        std::fs::remove_file(&path).expect("remove");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn two_planes_of_one_mapping_are_disjoint() {
        let path = temp_path("disjoint");
        let path_s = path.to_str().expect("utf-8 temp path");
        let map = Arc::new(KvMap::create(path_s, 6 * 8).expect("mapping"));
        let mut a = KvPlane::<i64>::mapped(Arc::clone(&map), 0, 3).expect("a");
        let mut b = KvPlane::<i64>::mapped(Arc::clone(&map), 3 * 8, 3).expect("b");
        a.copy_from_slice(&[1, 2, 3]);
        b.copy_from_slice(&[4, 5, 6]);
        assert_eq!(&a[..], &[1, 2, 3]);
        assert_eq!(&b[..], &[4, 5, 6]);
        drop((a, b, map));
        std::fs::remove_file(&path).expect("remove");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_mapped_plane_clone_is_owned_and_independent() {
        let path = temp_path("clone");
        let path_s = path.to_str().expect("utf-8 temp path");
        let map = Arc::new(KvMap::create(path_s, 2 * 8).expect("mapping"));
        let mut plane = KvPlane::<i64>::mapped(Arc::clone(&map), 0, 2).expect("plane");
        plane.copy_from_slice(&[11, 22]);
        let mut clone = plane.clone();
        clone[0] = 99;
        assert_eq!(&plane[..], &[11, 22]);
        assert_eq!(&clone[..], &[99, 22]);
        assert!(clone.mapped_bytes().is_none());
        drop((plane, map));
        std::fs::remove_file(&path).expect("remove");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_range_outside_the_mapping_is_refused() {
        let path = temp_path("range");
        let path_s = path.to_str().expect("utf-8 temp path");
        let map = Arc::new(KvMap::create(path_s, 8).expect("mapping"));
        assert!(matches!(
            KvPlane::<i64>::mapped(Arc::clone(&map), 8, 1),
            Err(KvFileError::BadRange)
        ));
        assert!(matches!(
            KvPlane::<i64>::mapped(Arc::clone(&map), 4, 1),
            Err(KvFileError::BadRange)
        ));
        assert!(matches!(
            KvPlane::<i64>::mapped(Arc::clone(&map), 0, usize::MAX),
            Err(KvFileError::Overflow)
        ));
        drop(map);
        std::fs::remove_file(&path).expect("remove");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_zero_length_mapping_is_refused() {
        let path = temp_path("empty");
        let path_s = path.to_str().expect("utf-8 temp path");
        assert!(matches!(KvMap::create(path_s, 0), Err(KvFileError::Empty)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_owned_plane_is_its_vector() {
        let mut plane = KvPlane::owned(vec![1i64, 2, 3]);
        plane[2] = 7;
        assert_eq!(&plane[..], &[1, 2, 7]);
        assert!(plane.mapped_bytes().is_none());
        let clone = plane.clone();
        assert_eq!(&clone[..], &[1, 2, 7]);
    }
}
