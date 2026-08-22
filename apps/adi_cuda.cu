// adi_cuda.cu -- 3-D ADI pentadiagonal solver, GPU version.
//
// Each ADI iteration performs three independent batch pentadiagonal solves
// (x, y, z) on a cube of side N.
//
// Timing.  The headline figure is the true end-to-end time of one iteration,
// not the sum of the three per-direction times, which under-reports by
// discarding launch overhead and inter-kernel gaps.  Two passes are run:
//   Pass A  the whole timed loop bracketed by one CUDA event pair and a CPU
//           wall clock, with nothing inserted between the direction solves.
//           This is the headline number.
//   Pass B  per-direction events give the x/y/z breakdown.  Kept separate
//           because the events perturb the stream slightly.
// The report prints both and their difference.
//
// Bandwidth.  Two figures per direction: BW uses an algorithm-aware array-pass
// model (kernels move very different amounts of scratch), so it reflects
// hardware efficiency; [vis] counts visible I/O only (7 arrays x N^3) and is
// the fair cross-algorithm figure.
//
// Usage:  ./adi_cuda [N] [niters] [precision]
//   N          grid size per dimension (default: 256)
//   niters     number of timed ADI iterations (default: 10)
//   precision  "double" or "float" (default: double)
//
// Algorithm selection is via PENTA_XALGO / PENTA_YZALGO (see README).

#include "pentadsolver.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

template <typename Float>
__global__ void fill_kernel(Float *arr, Float val, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

template <typename Float>
static void gpu_fill(Float *d, Float val, size_t n) {
    unsigned blocks = (unsigned)((n + 255) / 256);
    fill_kernel<<<blocks, 256>>>(d, val, n);
    check(cudaGetLastError(), "fill_kernel");
}

// ---------------------------------------------------------------------------
// Algorithm identification and DRAM traffic model
// ---------------------------------------------------------------------------
// "Array passes" = number of full N^3 arrays moved through DRAM by one solve.
// Used only to turn a measured time into an achieved-bandwidth figure.  The
// visible-I/O figure (always 7) is reported alongside and is the one to use
// when comparing different algorithms against each other.

static const char *env_or(const char *name, const char *fallback) {
    const char *v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : fallback;
}

// The strided directions are selected independently: PENTA_YALGO / PENTA_ZALGO
// take precedence, with PENTA_YZALGO applying to both as a fallback.  This
// mirrors strided_algo_selector() in the solver.
static const char *strided_selector(const char *specific_name) {
    const char *specific = std::getenv(specific_name);
    if (specific != nullptr && specific[0] != '\0') { return specific; }
    return env_or("PENTA_YZALGO", "auto");
}

// Resolve "auto" to the kernel the production dispatch actually selects, so
// the traffic model matches what really ran.
//
// An unrecognised selector falls through to the auto path inside the solver,
// so it must resolve here too, otherwise the banner and the CSV would name a
// kernel that did not run, and array_passes_x() would apply the wrong traffic
// model to it.
static bool is_known_x_algo(const char *a) {
    return std::strcmp(a, "naive")  == 0 || std::strcmp(a, "transpose")  == 0 ||
           std::strcmp(a, "thomas-pcr")  == 0 || std::strcmp(a, "shared-fact") == 0;
}

static const char *resolve_x_algo(const char *requested, bool is_fp32) {
    if (std::strcmp(requested, "auto") != 0 && is_known_x_algo(requested)) {
        return requested;
    }
    if (is_fp32) { return "thomas-pcr"; }
    // FP64 x is chosen from the device's FP64:FP32 ratio, Algorithm 3 wins
    // wherever FP64 is not crippled, Algorithm 2 where it is.  This mirrors
    // device_fp64_ratio_denom() in pentadsolver_gpsv_batch_x; keep in step.
    int dev = 0;
    int major = 0;
    int minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { return "transpose"; }
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    switch (major * 10 + minor) {
        case 86: case 87: case 89: return "transpose";  // 1/64 FP64
        default:                   return "thomas-pcr";  // 1/32 or better
    }
}

// Both strided directions auto-select Algorithm 3 in FP32 (measured wins:
// y 3.19 vs 4.36 ms, z 4.38 vs 4.52 ms).  FP64 stays on Algorithm 1, where
// the redundant arithmetic is compute-bound.  Algorithm 4 is opt-in.
static const char *resolve_strided_algo(const char *requested, bool is_fp32) {
    const bool known = std::strcmp(requested, "naive") == 0 ||
                       std::strcmp(requested, "thomas-pcr")  == 0 ||
                       std::strcmp(requested, "shared-fact") == 0;
    if (std::strcmp(requested, "auto") != 0 && known) { return requested; }
    return is_fp32 ? "thomas-pcr" : "naive";
}

static int array_passes_x(const char *algo) {
    if (std::strcmp(algo, "naive")  == 0) { return 7;  } // nominal only, see note below
    if (std::strcmp(algo, "transpose")  == 0) { return 27; } // 6 fwd-transposed + solve + 1 bwd
    if (std::strcmp(algo, "thomas-pcr")  == 0) { return 7;  } // register-resident, no scratch
    if (std::strcmp(algo, "shared-fact") == 0) { return 4;  } // b in + x2 wr + x2 rd + x out
    return 13;
}

static int array_passes_strided(const char *algo) {
    if (std::strcmp(algo, "shared-fact") == 0) { return 4; }
    if (std::strcmp(algo, "thomas-pcr")  == 0) { return 7; } // register-resident, no scratch
    return 13;  // legacy strided: 6 rd + 3 scratch wr + 3 scratch rd + 1 wr
}

// Algorithm 1 (naive) is uncoalesced: its NOMINAL traffic is 7 arrays but the
// EFFECTIVE traffic is far higher (each strided access pulls a full cache
// sector to use one element).  Its "% peak" is therefore not meaningful and
// is suppressed rather than reported as a fabricated number.
static bool traffic_model_is_meaningful(const char *algo) {
    return std::strcmp(algo, "naive") != 0;
}

// The solver records the kernel each direction ACTUALLY launched.  resolve_*()
// above can only predict, and a prediction is not always right: Algorithm 3
// declines a size whose template is not instantiated or whose shared tile does
// not fit, and the solver then falls back.  Everything after the warm-up,
// the traffic model, the report and the CSV, uses this, so a fallback is
// recorded as what ran rather than as what was asked for.  0 = x, 1 = y, 2 = z.
extern "C" const char *pentadsolver_kernel_that_ran(int dir);

// ---------------------------------------------------------------------------
// Measurement passes
// ---------------------------------------------------------------------------

struct AdiTimings {
    double e2e_wall_ms   = 0.0;  // per iteration, CPU wall clock (headline)
    double e2e_events_ms = 0.0;  // per iteration, GPU-side CUDA events
    double x_ms          = 0.0;  // per iteration, per-direction breakdown
    double y_ms          = 0.0;
    double z_ms          = 0.0;
};

template <typename Float>
static void run_one_iteration(pentadsolver_handle_t handle, Float *d_ds,
                              Float *d_dl, Float *d_d, Float *d_du,
                              Float *d_dw, Float *d_x, int *dims, int ndims,
                              void *d_buf) {
    for (int solvedim = 0; solvedim < 3; solvedim++) {
        pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims,
                                ndims, solvedim, d_buf);
    }
}

// Pass A, nothing inserted between the solves: the honest end-to-end cost.
template <typename Float>
static void measure_end_to_end(pentadsolver_handle_t handle, Float *d_ds,
                               Float *d_dl, Float *d_d, Float *d_du,
                               Float *d_dw, Float *d_x, int *dims, int ndims,
                               void *d_buf, int niters, AdiTimings &out) {
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    check(cudaDeviceSynchronize(), "pre-e2e sync");
    const auto wall_t0 = std::chrono::steady_clock::now();
    cudaEventRecord(ev_start);

    for (int it = 0; it < niters; it++) {
        run_one_iteration(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims,
                          ndims, d_buf);
    }

    cudaEventRecord(ev_stop);
    check(cudaEventSynchronize(ev_stop), "e2e event sync");
    const auto wall_t1 = std::chrono::steady_clock::now();

    float gpu_ms = 0.0F;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    out.e2e_events_ms = (double)gpu_ms / niters;
    out.e2e_wall_ms   = wall_ms / niters;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
}

// Pass B, per-direction events for the x/y/z breakdown.
template <typename Float>
static void measure_per_direction(pentadsolver_handle_t handle, Float *d_ds,
                                  Float *d_dl, Float *d_d, Float *d_du,
                                  Float *d_dw, Float *d_x, int *dims,
                                  int ndims, void *d_buf, int niters,
                                  AdiTimings &out) {
    cudaEvent_t ev_start[3];
    cudaEvent_t ev_stop[3];
    for (int d = 0; d < 3; d++) {
        cudaEventCreate(&ev_start[d]);
        cudaEventCreate(&ev_stop[d]);
    }

    double acc[3] = {0.0, 0.0, 0.0};

    check(cudaDeviceSynchronize(), "pre-breakdown sync");
    for (int it = 0; it < niters; it++) {
        for (int solvedim = 0; solvedim < 3; solvedim++) {
            cudaEventRecord(ev_start[solvedim]);
            pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x,
                                    dims, ndims, solvedim, d_buf);
            cudaEventRecord(ev_stop[solvedim]);
        }
        check(cudaEventSynchronize(ev_stop[2]), "breakdown event sync");

        for (int d = 0; d < 3; d++) {
            float ms = 0.0F;
            cudaEventElapsedTime(&ms, ev_start[d], ev_stop[d]);
            acc[d] += ms;
        }
    }

    out.x_ms = acc[0] / niters;
    out.y_ms = acc[1] / niters;
    out.z_ms = acc[2] / niters;

    for (int d = 0; d < 3; d++) {
        cudaEventDestroy(ev_start[d]);
        cudaEventDestroy(ev_stop[d]);
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

// Peak DRAM bandwidth used for the "% peak" figures.  Detected from the device
// so the number is meaningful on any GPU rather than hard-coded to one card.
//
// CAVEAT (important on some GPUs): the detected value is the P0 / spec memory
// clock, but CUDA compute workloads are placed in the P2 power state, which on
// several NVIDIA parts clocks memory LOWER than spec.  On a GTX 1080, spec is
// 5005 MHz (320 GB/s) but CUDA runs at 4513 MHz (289 GB/s), so "% of spec"
// understates efficiency by ~11%.  Check the clock actually in use with
//     nvidia-smi --query-gpu=pstate,clocks.mem --format=csv
// while the solver is running, and override via PENTA_PEAK_BW_GBS if it
// differs from spec.
static double detect_peak_bw() {
    const char *env = getenv("PENTA_PEAK_BW_GBS");
    if (env != nullptr) {
        const double v = atof(env);
        if (v > 0.0) { return v * 1e9; }
    }
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        int mem_khz = 0, bus_bits = 0;
        cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev);
        cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
        if (mem_khz > 0 && bus_bits > 0) {
            // DDR: two transfers per clock.
            return (double)mem_khz * 1e3 * 2.0 * ((double)bus_bits / 8.0);
        }
    }
    return 224.0e9;   // fallback: RTX 3050 GDDR6 peak
}

static void print_direction_line(const char *label, double ms, int passes,
                                 double elem_bytes, double peak_bw,
                                 bool model_meaningful) {
    const double secs   = ms * 1e-3;
    const double bw     = passes * elem_bytes / secs;
    const double bw_vis = 7.0 * elem_bytes / secs;

    if (model_meaningful) {
        printf("    %-11s %8.3f ms   BW: %6.1f GB/s (%4.1f%% peak, %2d passes)"
               "  [vis: %5.1f GB/s]\n",
               label, ms, bw / 1e9, bw / peak_bw * 100.0, passes, bw_vis / 1e9);
    } else {
        printf("    %-11s %8.3f ms   BW:  n/a (uncoalesced: effective traffic"
               " >> nominal)  [vis: %5.1f GB/s]\n",
               label, ms, bw_vis / 1e9);
    }
}

// ---------------------------------------------------------------------------
// Templated ADI run, identical logic/timing methodology for double & float
// ---------------------------------------------------------------------------

template <typename Float>
static void run_adi(int N, int NITERS, const char *precision_name) {
    // Full ADI loops before timing starts.  5 is enough on a GPU that reaches
    // its steady boost clock quickly, but NOT on every card: a GTX 1080 takes
    // ~350 ms to ramp from idle, so with 5 warm-up loops the end-to-end pass
    // (which runs first) is measured partly at low clocks while the
    // per-direction pass that follows is not, showing up as a spurious
    // "inter-kernel overhead" of up to 40%.  Raise it via PENTA_WARMUP when
    // the reported overhead shrinks as the iteration count grows, which is the
    // signature of a fixed startup cost rather than a real per-iteration gap.
    int NWARMUP = 5;
    if (const char *w = getenv("PENTA_WARMUP")) {
        const int v = atoi(w);
        if (v >= 0) { NWARMUP = v; }
    }
    const int NDIMS   = 3;
    const bool is_fp32 = (sizeof(Float) == 4);

    // Diagonally dominant coefficients: |d| = 5 > |ds|+|dl|+|du|+|dw| = 4
    const Float D_VAL = Float(5.0);

    size_t n_total = (size_t)N * N * N;
    size_t bytes   = n_total * sizeof(Float);
    int    dims[3] = {N, N, N};

    const char *req_x = env_or("PENTA_XALGO", "auto");
    const char *req_y = strided_selector("PENTA_YALGO");
    const char *req_z = strided_selector("PENTA_ZALGO");
    const char *eff_x = resolve_x_algo(req_x, is_fp32);
    const char *eff_y = resolve_strided_algo(req_y, is_fp32);
    const char *eff_z = resolve_strided_algo(req_z, is_fp32);

    // The device banner is QUERIED, never hardcoded: this binary is built and
    // run on more than one GPU (sm_86 and sm_61), and a stale hardcoded name
    // silently mislabels every result file produced on the other machine.
    cudaDeviceProp dprop{};
    int            dev_id = 0;
    cudaGetDevice(&dev_id);
    cudaGetDeviceProperties(&dprop, dev_id);
    printf("=========================================================\n");
    printf("Pentadiagonal ADI, GPU  (%s, sm_%d%d, CUDA %d.%d)\n",
           dprop.name, dprop.major, dprop.minor,
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("Grid: %d x %d x %d   (%zu elements)\n", N, N, N, n_total);
    printf("Precision: %-6s  Warmup: %d iter   Timed: %d iter\n",
           precision_name, NWARMUP, NITERS);
    printf("Algorithm: x=%s(%s)  y=%s(%s)  z=%s(%s)\n",
           req_x, eff_x, req_y, eff_y, req_z, eff_z);
    printf("=========================================================\n\n");

    // ------------------------------------------------------------------
    // Allocate and initialise device arrays
    // ------------------------------------------------------------------
    Float *d_ds, *d_dl, *d_d, *d_du, *d_dw, *d_x;
    check(cudaMalloc(&d_ds, bytes), "malloc ds");
    check(cudaMalloc(&d_dl, bytes), "malloc dl");
    check(cudaMalloc(&d_d,  bytes), "malloc d");
    check(cudaMalloc(&d_du, bytes), "malloc du");
    check(cudaMalloc(&d_dw, bytes), "malloc dw");
    check(cudaMalloc(&d_x,  bytes), "malloc x");

    gpu_fill(d_ds, Float(1.0), n_total);
    gpu_fill(d_dl, Float(1.0), n_total);
    gpu_fill(d_d,  D_VAL,      n_total);
    gpu_fill(d_du, Float(1.0), n_total);
    gpu_fill(d_dw, Float(1.0), n_total);
    gpu_fill(d_x,  Float(1.0), n_total);
    check(cudaDeviceSynchronize(), "init sync");

    // ------------------------------------------------------------------
    // Allocate scratch buffer (maximum across all three directions)
    // ------------------------------------------------------------------
    pentadsolver_handle_t handle{};
    pentadsolver_create(&handle, nullptr, 0, nullptr);

    size_t buf_x = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, 0);
    size_t buf_y = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, 1);
    size_t buf_z = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, 2);
    size_t buf_size = buf_x > buf_y ? buf_x : buf_y;
    if (buf_z > buf_size) buf_size = buf_z;

    void *d_buf = nullptr;
    if (buf_size > 0) check(cudaMalloc(&d_buf, buf_size), "malloc buf");

    printf("Scratch buffer: %.1f MB\n\n", (double)buf_size / (1024.0 * 1024.0));

    // Warm-up: bring the GPU to boost clock before timing starts.
    //
    // Bounded by wall time, not iteration count: what has to finish is a clock
    // ramp measured in milliseconds, and a fixed count gives a fast kernel far
    // less ramp than a slow one.  That asymmetry showed up as a spurious 34%
    // "inter-kernel overhead", Pass A absorbing the ramp where Pass B did not.
    //
    // NWARMUP is the floor; PENTA_WARMUP_MS is the wall-time target that
    // governs (0 restores pure count-based behaviour).  The 1500 ms default
    // comes from a measured knee: on a GTX 1080 at 256^3 FP64 the reported
    // end-to-end is 10.21 ms at 900 ms of warm-up, 8.43 at 1200, and 8.38 flat
    // from 1500 to 3000.
    double warmup_ms_target = 1500.0;
    if (const char *w = getenv("PENTA_WARMUP_MS")) {
        const double v = atof(w);
        if (v >= 0.0) { warmup_ms_target = v; }
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        int        it = 0;
        double     elapsed = 0.0;
        while (it < NWARMUP || elapsed < warmup_ms_target) {
            run_one_iteration(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims,
                              NDIMS, d_buf);
            check(cudaDeviceSynchronize(), "warmup sync");
            it++;
            elapsed = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
        }
        printf("Warm-up: %d iterations, %.0f ms wall (target %.0f ms)\n\n", it,
               elapsed, warmup_ms_target);
    }

    // The warm-up has now exercised every direction, so the solver's record of
    // what it launched is populated.  Adopt it: from here on eff_* is a
    // measured fact rather than a prediction.
    eff_x = pentadsolver_kernel_that_ran(0);
    eff_y = pentadsolver_kernel_that_ran(1);
    eff_z = pentadsolver_kernel_that_ran(2);
    printf("Kernels that actually ran: x=%s  y=%s  z=%s\n", eff_x, eff_y,
           eff_z);
    {
        // A request that was not honoured is a silent measurement error unless
        // it is said out loud, the run still produces a plausible-looking
        // number, just for a different kernel.
        const char *req[3] = {req_x, req_y, req_z};
        const char *got[3] = {eff_x, eff_y, eff_z};
        const char  dir[3] = {'x', 'y', 'z'};
        for (int i = 0; i < 3; i++) {
            if (std::strcmp(req[i], "auto") != 0 &&
                std::strcmp(req[i], got[i]) != 0) {
                printf("  !! WARNING: %c requested '%s' but ran '%s' "
                       "(unsupported at this size/precision -- fell back)\n",
                       dir[i], req[i], got[i]);
            }
        }
    }
    printf("\n");

    // ------------------------------------------------------------------
    // Measure
    // ------------------------------------------------------------------
    AdiTimings t;
    measure_end_to_end(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS,
                       d_buf, NITERS, t);
    measure_per_direction(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims,
                          NDIMS, d_buf, NITERS, t);

    // ------------------------------------------------------------------
    // Report
    // ------------------------------------------------------------------
    const double elem_bytes = (double)n_total * sizeof(Float);
    const double peak_bw    = detect_peak_bw();
    const double sum_parts  = t.x_ms + t.y_ms + t.z_ms;
    const double overhead   = t.e2e_events_ms - sum_parts;
    const double overhead_pct =
        (t.e2e_events_ms > 0.0) ? (overhead / t.e2e_events_ms * 100.0) : 0.0;

    printf("GPU timings (avg over %d iterations):\n", NITERS);
    printf("  ('%% peak' is against %.1f GB/s%s)\n\n", peak_bw / 1e9,
           getenv("PENTA_PEAK_BW_GBS") != nullptr ? ", set by PENTA_PEAK_BW_GBS"
                                                  : ", detected from device");
    printf("  END-TO-END (one full ADI iteration, x+y+z back-to-back):\n");
    printf("    %-11s %8.3f ms   <-- headline (incl. launch overhead)\n",
           "wall clock:", t.e2e_wall_ms);
    printf("    %-11s %8.3f ms   (GPU-side, CUDA events)\n",
           "gpu events:", t.e2e_events_ms);
    printf("\n");
    printf("  PER-DIRECTION BREAKDOWN (separate instrumented pass):\n");
    print_direction_line("pentad_x:", t.x_ms, array_passes_x(eff_x),
                         elem_bytes, peak_bw,
                         traffic_model_is_meaningful(eff_x));
    print_direction_line("pentad_y:", t.y_ms, array_passes_strided(eff_y),
                         elem_bytes, peak_bw, true);
    print_direction_line("pentad_z:", t.z_ms, array_passes_strided(eff_z),
                         elem_bytes, peak_bw, true);
    printf("    %-11s %8.3f ms\n", "sum(x+y+z):", sum_parts);
    printf("\n");
    printf("  Inter-kernel overhead (end-to-end - sum of parts): %.3f ms"
           " (%.1f%% of end-to-end)\n", overhead, overhead_pct);
    printf("  NOTE: sum(x+y+z) is NOT the iteration cost, it omits the gaps\n"
           "        above.  Quote the end-to-end wall clock as the total.\n");

    // Kept for backward compatibility with scripts that grep this token; it
    // now reports the TRUE end-to-end time, not the sum of the parts.
    printf("  %-12s %8.3f ms   (end-to-end wall clock)\n", "total/iter:",
           t.e2e_wall_ms);

    // Machine-readable summary for the algorithm sweep script.  Each
    // direction reports the selector it was given and the kernel that ran
    // (they differ only when the selector is "auto").
    printf("\nCSV,%s,%s,%s,%s,%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n",
           req_x, eff_x, req_y, eff_y, req_z, eff_z, precision_name, N,
           t.e2e_wall_ms, t.e2e_events_ms, t.x_ms, t.y_ms, t.z_ms, sum_parts,
           overhead_pct);

    // CPU reference, per precision.  NOTE: these were measured with the
    // OpenMP+AVX2 CPU library at its saturated multi-core throughput
    // (6 threads; scaling saturates at T=4, the i5-9500 is memory-bound
    // at ~20 GB/s STREAM triad).  These are per-direction figures from a
    // separate tool, so their "total" is a sum and carries the same caveat
    // as above; it is shown only as an order-of-magnitude reference.
    // These constants were measured at N=256 on cobra-01 (Intel i5-9500).
    // They are NOT valid at other sizes and NOT valid on another machine, so
    // the comparison is suppressed unless this run matches the size they were
    // taken at.  (Previously they were printed for every N, which produced a
    // meaningless "68.79x speedup" at N=64.)
    if (N != 256) {
        printf("\nCPU baseline: not shown, the built-in reference was measured"
               " at 256^3 on cobra-01\n  (Intel i5-9500) and does not apply to"
               " %d^3.  Measure this machine with apps/adi_cpu.\n", N);
        pentadsolver_destroy(&handle);
        cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
        cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
        if (d_buf != nullptr) { cudaFree(d_buf); }
        return;
    }

    double cpu_x, cpu_y, cpu_z;
    if (sizeof(Float) == 8) {
        cpu_x = 36.10; cpu_y = 62.42; cpu_z = 151.21;   // FP64, 6 threads
    } else {
        cpu_x = 18.16; cpu_y = 25.01; cpu_z = 76.17;    // FP32, 6 threads
    }
    const double cpu_total = cpu_x + cpu_y + cpu_z;

    printf("\nCPU baseline (%s, %d^3, OpenMP+AVX2 Thomas, 6 threads, measured"
           " on cobra-01 i5-9500, reference only):\n", precision_name, N);
    printf("  pentad_x:  %6.2f ms\n", cpu_x);
    printf("  pentad_y:  %6.2f ms\n", cpu_y);
    printf("  pentad_z:  %6.2f ms\n", cpu_z);
    printf("  sum:       %6.2f ms\n", cpu_total);

    printf("\nGPU speedup vs CPU:\n");
    printf("  pentad_x:  %5.2fx\n", cpu_x / t.x_ms);
    printf("  pentad_y:  %5.2fx\n", cpu_y / t.y_ms);
    printf("  pentad_z:  %5.2fx\n", cpu_z / t.z_ms);
    printf("  overall:   %5.2fx  (CPU sum vs GPU end-to-end)\n",
           cpu_total / t.e2e_wall_ms);

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
    cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
    if (d_buf) cudaFree(d_buf);
    pentadsolver_destroy(&handle);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    const int N       = (argc > 1) ? atoi(argv[1]) : 256;
    const int NITERS  = (argc > 2) ? atoi(argv[2]) : 10;
    const char *prec  = (argc > 3) ? argv[3] : "double";

    if (std::strcmp(prec, "float") == 0) {
        run_adi<float>(N, NITERS, "float");
    } else if (std::strcmp(prec, "double") == 0) {
        run_adi<double>(N, NITERS, "double");
    } else {
        fprintf(stderr, "Unknown precision '%s' (expected 'double' or 'float')\n", prec);
        return 1;
    }
    return 0;
}
