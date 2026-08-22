// adi_cpu.cpp: 3-D ADI pentadiagonal solver, CPU version
//
// Methodology mirror of apps/adi_cuda.cu so CPU and GPU end-to-end numbers
// are directly comparable: same grid, same diagonally-dominant coefficients,
// same warmup + timed-iteration structure, per-direction timing.  The CPU
// library (src/cpu/singlenode/pentad_cpu.cpp) is OpenMP + AVX2-SIMD
// parallel; thread count is controlled at runtime via OMP_NUM_THREADS.
//
// Usage:  ./adi_cpu [N] [niters] [precision]
//   N          grid size per dimension (default: 256)
//   niters     number of timed ADI iterations (default: 10)
//   precision  "double" or "float" (default: double)

#include "pentadsolver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

template <typename Float>
static void run_adi(int N, int NITERS, const char *precision_name) {
    const int NWARMUP = 2;
    const int NDIMS   = 3;

    // Diagonally dominant coefficients: |d| = 5 > |ds|+|dl|+|du|+|dw| = 4
    // (identical to adi_cuda.cu)
    size_t n_total = (size_t)N * N * N;
    int    dims[3] = {N, N, N};

    printf("=========================================================\n");
    printf("Pentadiagonal ADI, CPU  (OpenMP + AVX2 SIMD)\n");
    printf("Grid: %d x %d x %d   (%zu elements)\n", N, N, N, n_total);
    printf("Precision: %-6s  Threads: %d   Warmup: %d iter   Timed: %d iter\n",
           precision_name, omp_get_max_threads(), NWARMUP, NITERS);
    printf("=========================================================\n\n");

    std::vector<Float> ds(n_total, Float(1.0));
    std::vector<Float> dl(n_total, Float(1.0));
    std::vector<Float> d (n_total, Float(5.0));
    std::vector<Float> du(n_total, Float(1.0));
    std::vector<Float> dw(n_total, Float(1.0));
    std::vector<Float> x (n_total, Float(1.0));

    pentadsolver_handle_t handle{};

    // Warmup (also faults in all pages before timing)
    for (int it = 0; it < NWARMUP; it++) {
        for (int dim = 0; dim < 3; dim++) {
            pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                    du.data(), dw.data(), x.data(),
                                    dims, NDIMS, dim, nullptr);
        }
    }

    double t_x = 0.0, t_y = 0.0, t_z = 0.0;
    using clk = std::chrono::steady_clock;

    for (int it = 0; it < NITERS; it++) {
        auto t0 = clk::now();
        pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                du.data(), dw.data(), x.data(),
                                dims, NDIMS, 0, nullptr);
        auto t1 = clk::now();
        pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                du.data(), dw.data(), x.data(),
                                dims, NDIMS, 1, nullptr);
        auto t2 = clk::now();
        pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                du.data(), dw.data(), x.data(),
                                dims, NDIMS, 2, nullptr);
        auto t3 = clk::now();

        t_x += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_y += std::chrono::duration<double, std::milli>(t2 - t1).count();
        t_z += std::chrono::duration<double, std::milli>(t3 - t2).count();
    }

    double avg_x = t_x / NITERS, avg_y = t_y / NITERS, avg_z = t_z / NITERS;
    double avg_total = avg_x + avg_y + avg_z;

    // Visible I/O per solve: 7 arrays (ds dl d du dw x_in x_out)
    double elem_bytes = (double)n_total * sizeof(Float);
    double bw_vis     = 7.0 * elem_bytes;

    printf("CPU timings (avg over %d iterations, steady_clock):\n", NITERS);
    printf("  %-12s %8.3f ms   [vis BW: %5.1f GB/s]\n", "pentad_x:", avg_x,
           bw_vis / (avg_x * 1e-3) / 1e9);
    printf("  %-12s %8.3f ms   [vis BW: %5.1f GB/s]\n", "pentad_y:", avg_y,
           bw_vis / (avg_y * 1e-3) / 1e9);
    printf("  %-12s %8.3f ms   [vis BW: %5.1f GB/s]\n", "pentad_z:", avg_z,
           bw_vis / (avg_z * 1e-3) / 1e9);
    printf("  %-12s %8.3f ms\n", "total/iter:", avg_total);
    printf("  (vis BW = 7-array visible I/O; compare against STREAM triad peak)\n");

    // Machine-readable summary, mirroring the CSV line in adi_cuda.cu so a
    // thread sweep can be scripted without parsing the pretty output.  The
    // thread count is recorded per row because it is set externally via
    // OMP_NUM_THREADS and is otherwise unrecoverable from the numbers.
    printf("\nCSV,%s,%d,%d,%.4f,%.4f,%.4f,%.4f\n", precision_name, N,
           omp_get_max_threads(), avg_x, avg_y, avg_z, avg_total);
}

int main(int argc, char **argv) {
    const int   N      = (argc > 1) ? atoi(argv[1]) : 256;
    const int   NITERS = (argc > 2) ? atoi(argv[2]) : 10;
    const char *prec   = (argc > 3) ? argv[3] : "double";

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
