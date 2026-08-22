// roofline_probe.cu: measure a GPU's FP64 issue rate and a kernel's memory
// floor WITHOUT any hardware counters.
//
// Method ("arithmetic dial"): run a kernel with exactly the memory traffic of
// the pentadiagonal x-solve (6 arrays read + 1 written, N^3 doubles) and K
// injected independent FP64 FMAs per element.  Sweep K.
//
//   time(K) = memory_floor + K * seconds_per_FMA_per_element
//
// The INTERCEPT is the achievable memory time for that exact traffic, the
// number the array-pass model only estimates.  The SLOPE is the machine's FP64
// issue rate, the 0.2360 ms/slot constant the SASS cost model assumes.
// Neither requires ncu.
//
// Four independent accumulators keep the FMAs issue-bound rather than
// latency-bound, matching the regime the real kernels run in.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

static void ck(cudaError_t e, const char *m) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

template <int K>
__global__ void probe(const double *__restrict__ a0, const double *__restrict__ a1,
                      const double *__restrict__ a2, const double *__restrict__ a3,
                      const double *__restrict__ a4, const double *__restrict__ a5,
                      double *__restrict__ out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double v0 = a0[i], v1 = a1[i], v2 = a2[i], v3 = a3[i], v4 = a4[i], v5 = a5[i];
    // Seed with all SIX inputs.  If v4/v5 only appeared inside the K-loop the
    // compiler would delete their loads at K=0 and the "memory floor" would be
    // measured on 5 arrays instead of 7, which showed up as an impossible
    // 293 GB/s, above the card's peak.  Two extra adds is a cheap guarantee.
    double s0 = v0 + v4, s1 = v1 + v5, s2 = v2, s3 = v3;
    const double c = 1.0000000001;
#pragma unroll
    for (int k = 0; k < K / 4; k++) {
        s0 = fma(s0, c, v4);
        s1 = fma(s1, c, v5);
        s2 = fma(s2, c, v0);
        s3 = fma(s3, c, v1);
    }
    out[i] = s0 + s1 + s2 + s3;
}

// Warm-up must be measured in WALL TIME, not launch count.  A fixed count is
// wrong because a slow kernel warms up for longer than a fast one, which makes
// the sweep non-monotonic: on a GTX 1080 (~350 ms to reach boost clock) a
// 5-launch warm-up left every point on this curve at a different clock, and
// K=64 came out FASTER than K=32.
static const double WARM_MS = 700.0;

template <int K>
static double run(const double *const *a, double *out, size_t n, int reps) {
    dim3 blk(256), grd((unsigned)((n + 255) / 256));
    auto wall0 = std::chrono::steady_clock::now();
    do {
        for (int w = 0; w < 4; w++)
            probe<K><<<grd, blk>>>(a[0], a[1], a[2], a[3], a[4], a[5], out, n);
        ck(cudaDeviceSynchronize(), "warmup");
    } while (std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - wall0).count() < WARM_MS);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < reps; r++)
        probe<K><<<grd, blk>>>(a[0], a[1], a[2], a[3], a[4], a[5], out, n);
    cudaEventRecord(t1);
    ck(cudaEventSynchronize(t1), "sync");
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / reps;
}

int main(int argc, char **argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 256;
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;
    const size_t n = (size_t)N * N * N;
    const double bytes = 7.0 * (double)n * sizeof(double);   // 6 in + 1 out

    double *buf[7];
    for (int i = 0; i < 7; i++) {
        ck(cudaMalloc(&buf[i], n * sizeof(double)), "malloc");
        ck(cudaMemset(buf[i], 0x3f, n * sizeof(double)), "memset");
    }
    const double *a[6] = {buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]};

    printf("N=%d  elements=%zu  traffic=%.1f MB (7 arrays)  reps=%d\n\n",
           N, n, bytes / 1e6, reps);
    printf("  %6s  %10s  %12s\n", "K", "time (ms)", "eff GB/s");
    printf("  %6s  %10s  %12s\n", "------", "----------", "------------");

    const int  Ks[] = {0, 8, 16, 32, 64, 128, 256};
    double     ts[7];
    ts[0] = run<0>  (a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 0,   ts[0], bytes / (ts[0] * 1e-3) / 1e9);
    ts[1] = run<8>  (a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 8,   ts[1], bytes / (ts[1] * 1e-3) / 1e9);
    ts[2] = run<16> (a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 16,  ts[2], bytes / (ts[2] * 1e-3) / 1e9);
    ts[3] = run<32> (a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 32,  ts[3], bytes / (ts[3] * 1e-3) / 1e9);
    ts[4] = run<64> (a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 64,  ts[4], bytes / (ts[4] * 1e-3) / 1e9);
    ts[5] = run<128>(a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 128, ts[5], bytes / (ts[5] * 1e-3) / 1e9);
    ts[6] = run<256>(a, buf[6], n, reps); printf("  %6d  %10.4f  %12.1f\n", 256, ts[6], bytes / (ts[6] * 1e-3) / 1e9);

    // Least-squares fit over the compute-bound tail (K >= 32), where the
    // memory time is fully hidden and the line is pure arithmetic.
    int i0 = 3, cnt = 4;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = i0; i < i0 + cnt; i++) { double x = Ks[i], y = ts[i];
        sx += x; sy += y; sxx += x * x; sxy += x * y; }
    double slope = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
    double icpt  = (sy - slope * sx) / cnt;

    printf("\n  fit over K>=32:  time_ms = %.4f + %.6f * K\n", icpt, slope);
    printf("  --> FP64 issue rate      : %.6f ms per (FMA per element)\n", slope);
    printf("  --> memory floor (K=0)   : %.4f ms measured, %.4f ms extrapolated\n", ts[0], icpt);
    printf("  --> achievable bandwidth : %.1f GB/s on 7-array traffic\n", bytes / (ts[0] * 1e-3) / 1e9);

    for (int i = 0; i < 7; i++) cudaFree(buf[i]);
    return 0;
}
