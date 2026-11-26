// Thin C++ binding between the official MLPerf LoadGen and the Rust SUT (examples/mlperf_sut.rs).
// No policy lives here: it forwards IssueQuery to Rust, forwards completions back, and builds
// TestSettings from the official mlperf.conf + user.conf exactly as the reference harness does.
#include <cstdint>
#include <string>
#include <vector>
#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

extern "C" {
typedef void (*issue_fn)(void*, const uint64_t* ids, const uint64_t* idx, size_t n);

struct Sut : mlperf::SystemUnderTest {
  std::string name = "sanguine-engine-rust";
  void* ctx; issue_fn fn;
  const std::string& Name() override { return name; }
  void IssueQuery(const std::vector<mlperf::QuerySample>& s) override {
    std::vector<uint64_t> ids, idx;
    for (auto& q : s) { ids.push_back(q.id); idx.push_back(q.index); }
    fn(ctx, ids.data(), idx.data(), ids.size());
  }
  void FlushQueries() override {}
};

struct Qsl : mlperf::QuerySampleLibrary {
  std::string name = "cnn_dailymail";
  size_t total, perf;
  const std::string& Name() override { return name; }
  size_t TotalSampleCount() override { return total; }
  size_t PerformanceSampleCount() override { return perf; }
  void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>&) override {}
  void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>&) override {}
};

void mlperf_complete(uint64_t id, const void* data, size_t bytes, int64_t ntok, int first) {
  mlperf::QuerySampleResponse r{id, (uintptr_t)data, bytes, ntok};
  if (first) mlperf::FirstTokenComplete(&r, 1); else mlperf::QuerySamplesComplete(&r, 1);
}

// mode: 0 performance, 1 accuracy. scenario: "Offline" | "Server".
// min_queries/min_ms/target_qps override the official conf ONLY when > 0 (subset runs).
int mlperf_run(void* ctx, issue_fn fn, size_t total, const char* scenario, int mode,
               const char* mlperf_conf, const char* user_conf, const char* model,
               const char* logdir, uint64_t min_queries, uint64_t min_ms, double target_qps) {
  Sut sut; sut.ctx = ctx; sut.fn = fn;
  Qsl qsl; qsl.total = total; qsl.perf = total;
  mlperf::TestSettings ts;
  std::string sc = scenario;
  ts.scenario = sc == "Server" ? mlperf::TestScenario::Server : mlperf::TestScenario::Offline;
  ts.mode = mode ? mlperf::TestMode::AccuracyOnly : mlperf::TestMode::PerformanceOnly;
  if (ts.FromConfig(mlperf_conf, model, scenario, 1) != 0) return 10;
  if (ts.FromConfig(user_conf, model, scenario, 2) != 0) return 11;
  ts.mode = mode ? mlperf::TestMode::AccuracyOnly : mlperf::TestMode::PerformanceOnly;
  if (min_queries) { ts.min_query_count = min_queries; ts.max_query_count = min_queries; }
  if (min_ms) ts.min_duration_ms = min_ms;
  if (target_qps > 0) ts.server_target_qps = target_qps;
  ts.performance_sample_count_override = total;
  mlperf::LogSettings ls;
  ls.log_output.outdir = logdir;
  ls.log_output.copy_summary_to_stdout = true;
  ls.enable_trace = false;
  mlperf::StartTest(&sut, &qsl, ts, ls);
  return 0;
}
}
