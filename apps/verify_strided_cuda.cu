// verify_strided_cuda.cu: correctness validator for the strided (y, z)
// direction kernels.
//
// The fixed Catch2 suite only exercises the default, which is the naive
// strided kernel for y and z.  Any other strided kernel therefore needs its own
// check before its timings can be trusted.
//
// For the requested direction this tool:
//   1. builds a random, strictly diagonally dominant pentadiagonal system on
//      every line of an N^3 grid (so the system is well conditioned and the
//      solve is unique),
//   2. solves it with the naive kernel (PENTA_YALGO/PENTA_ZALGO=naive),
//   3. re-uploads the identical RHS and solves with the kernel under test,
//   4. reports, over the FULL grid:
//        - the backward residual max|A*x - b| for each solve, and
//        - max|x_test - x_naive|.
//   The full-grid comparison is what catches system-mixing bugs (a kernel
//   that solves each line correctly but attributes lines to the wrong system
//   still shows a small residual per line yet a large cross-kernel diff).
//
// Usage: ./verify_strided_cuda [N] [double|float] [dir 1|2] [algo]
//   N     grid size per dimension          (default 256)
//   dir   1 = y (middle, stride N), 2 = z (outermost, stride N^2)  (default 1)
//   algo  kernel under test: thomas-pcr                        (default thomas-pcr)

#include "pentadsolver.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Index of element `i` along `dir` for the line identified by (a, b), matching
// the solver's own layout:  dir 1 (y): idx = a + i*N + b*N*N
//                           dir 2 (z): idx = a + b*N + i*N*N
static inline size_t line_index(int dir, int N, int a, int b, int i) {
    const size_t n  = (size_t)N;
    if (dir == 1) { return (size_t)a + (size_t)i * n + (size_t)b * n * n; }
    return (size_t)a + (size_t)b * n + (size_t)i * n * n;
}

template <typename Float>
static int run(int N, int dir, const char *algo) {
    const size_t n_total = (size_t)N * N * N;
    const size_t bytes   = n_total * sizeof(Float);
    const char  *dir_env = (dir == 1) ? "PENTA_YALGO" : "PENTA_ZALGO";
    const char  *dir_name = (dir == 1) ? "y (middle)" : "z (outermost)";

    printf("[verify_strided] N=%d prec=%s dir=%s algo=%s\n", N,
           sizeof(Float) == 8 ? "double" : "float", dir_name, algo);

    // ---------------------------------------------------------------
    // Build a random diagonally dominant system on the host
    // ---------------------------------------------------------------
    std::vector<Float> h_ds(n_total), h_dl(n_total), h_d(n_total);
    std::vector<Float> h_du(n_total), h_dw(n_total), h_b(n_total);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> off(-1.0, 1.0);
    std::uniform_real_distribution<double> rhs(-1.0, 1.0);

    for (size_t i = 0; i < n_total; i++) {
        const double a = off(rng), b = off(rng), c = off(rng), e = off(rng);
        h_ds[i] = (Float)a;
        h_dl[i] = (Float)b;
        h_du[i] = (Float)c;
        h_dw[i] = (Float)e;
        // strict diagonal dominance: |d| > |ds|+|dl|+|du|+|dw|
        h_d[i]  = (Float)(std::fabs(a) + std::fabs(b) + std::fabs(c) +
                          std::fabs(e) + 1.0);
        h_b[i]  = (Float)rhs(rng);
    }

    // Zero the coupling coefficients that fall outside each line, exactly as
    // the solver's data convention requires (first two / last two rows).
    const int n_lines = N;
    for (int a = 0; a < n_lines; a++) {
        for (int b = 0; b < n_lines; b++) {
            h_ds[line_index(dir, N, a, b, 0)] = Float(0);
            h_ds[line_index(dir, N, a, b, 1)] = Float(0);
            h_dl[line_index(dir, N, a, b, 0)] = Float(0);
            h_du[line_index(dir, N, a, b, N - 1)] = Float(0);
            h_dw[line_index(dir, N, a, b, N - 2)] = Float(0);
            h_dw[line_index(dir, N, a, b, N - 1)] = Float(0);
        }
    }

    // ---------------------------------------------------------------
    // Device buffers
    // ---------------------------------------------------------------
    Float *d_ds, *d_dl, *d_d, *d_du, *d_dw, *d_x;
    check(cudaMalloc(&d_ds, bytes), "malloc ds");
    check(cudaMalloc(&d_dl, bytes), "malloc dl");
    check(cudaMalloc(&d_d,  bytes), "malloc d");
    check(cudaMalloc(&d_du, bytes), "malloc du");
    check(cudaMalloc(&d_dw, bytes), "malloc dw");
    check(cudaMalloc(&d_x,  bytes), "malloc x");

    check(cudaMemcpy(d_ds, h_ds.data(), bytes, cudaMemcpyHostToDevice), "cp ds");
    check(cudaMemcpy(d_dl, h_dl.data(), bytes, cudaMemcpyHostToDevice), "cp dl");
    check(cudaMemcpy(d_d,  h_d.data(),  bytes, cudaMemcpyHostToDevice), "cp d");
    check(cudaMemcpy(d_du, h_du.data(), bytes, cudaMemcpyHostToDevice), "cp du");
    check(cudaMemcpy(d_dw, h_dw.data(), bytes, cudaMemcpyHostToDevice), "cp dw");

    int dims[3] = {N, N, N};
    pentadsolver_handle_t handle{};
    pentadsolver_create(&handle, nullptr, 0, nullptr);

    size_t buf_size = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, 3, dir);
    void *d_buf = nullptr;
    if (buf_size > 0) check(cudaMalloc(&d_buf, buf_size), "malloc buf");

    auto solve_with = [&](const char *sel, std::vector<Float> &out) {
        setenv(dir_env, sel, 1);
        check(cudaMemcpy(d_x, h_b.data(), bytes, cudaMemcpyHostToDevice), "cp b");
        pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims,
                                3, dir, d_buf);
        check(cudaDeviceSynchronize(), "solve sync");
        out.resize(n_total);
        check(cudaMemcpy(out.data(), d_x, bytes, cudaMemcpyDeviceToHost), "cp x");
        unsetenv(dir_env);
    };

    std::vector<Float> x_naive, x_test;
    solve_with("naive", x_naive);
    solve_with(algo,     x_test);

    // ---------------------------------------------------------------
    // Backward residual max|A*x - b| over the full grid, plus cross-kernel diff
    // ---------------------------------------------------------------
    auto residual = [&](const std::vector<Float> &x) {
        double worst = 0.0;
        for (int a = 0; a < n_lines; a++) {
            for (int b = 0; b < n_lines; b++) {
                for (int i = 0; i < N; i++) {
                    const size_t idx = line_index(dir, N, a, b, i);
                    double acc = (double)h_d[idx] * (double)x[idx];
                    if (i >= 2) acc += (double)h_ds[idx] *
                                       (double)x[line_index(dir, N, a, b, i - 2)];
                    if (i >= 1) acc += (double)h_dl[idx] *
                                       (double)x[line_index(dir, N, a, b, i - 1)];
                    if (i + 1 < N) acc += (double)h_du[idx] *
                                          (double)x[line_index(dir, N, a, b, i + 1)];
                    if (i + 2 < N) acc += (double)h_dw[idx] *
                                          (double)x[line_index(dir, N, a, b, i + 2)];
                    const double r = std::fabs(acc - (double)h_b[idx]);
                    if (r > worst) worst = r;
                }
            }
        }
        return worst;
    };

    const double r_naive = residual(x_naive);
    const double r_test   = residual(x_test);

    double max_diff = 0.0;
    for (size_t i = 0; i < n_total; i++) {
        const double diff = std::fabs((double)x_test[i] - (double)x_naive[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  backward resid naive = %.3e\n", r_naive);
    printf("  backward resid %-6s = %.3e\n", algo, r_test);
    printf("  max|%s - naive|%*s= %.3e   (full grid, %zu elements)\n",
           algo, (int)(6 - strlen(algo)) > 0 ? (int)(6 - strlen(algo)) : 0, "",
           max_diff, n_total);

    // Tolerances: round-off level for the precision in use, with headroom for
    // the different summation orders the two kernels use.
    const double tol = (sizeof(Float) == 8) ? 1e-10 : 1e-3;
    const bool   ok  = (r_test < tol) && (max_diff < tol);
    printf("  %s (tolerance %.0e)\n", ok ? "PASS" : "FAIL", tol);

    cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
    cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
    if (d_buf) cudaFree(d_buf);
    pentadsolver_destroy(&handle);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const int   N    = (argc > 1) ? atoi(argv[1]) : 256;
    const char *prec = (argc > 2) ? argv[2] : "double";
    const int   dir  = (argc > 3) ? atoi(argv[3]) : 1;
    const char *algo = (argc > 4) ? argv[4] : "thomas-pcr";

    if (dir != 1 && dir != 2) {
        fprintf(stderr, "dir must be 1 (y) or 2 (z)\n");
        return 1;
    }
    // This tool gives every line its own random coefficients, which is the case
    // shared-fact is not defined for: it factorises system 0 once and reuses it
    // for all of them.  Comparing the two here measures the problem being wrong
    // for the kernel, not the kernel being wrong.  verify_shared_fact_cuda
    // builds the shared-coefficient case and is the tool for that kernel.
    if (std::strcmp(algo, "shared-fact") == 0) {
        fprintf(stderr,
                "verify_strided_cuda cannot validate shared-fact: it builds "
                "per-line random coefficients,\nwhich shared-fact does not "
                "solve.  Use verify_shared_fact_cuda %d %s %d instead.\n",
                N, prec, dir);
        return 2;
    }
    if (std::strcmp(algo, "thomas-pcr") != 0) {
        fprintf(stderr, "algo must be thomas-pcr\n");
        return 1;
    }
    if (std::strcmp(prec, "float") == 0)  { return run<float>(N, dir, algo); }
    if (std::strcmp(prec, "double") == 0) { return run<double>(N, dir, algo); }
    fprintf(stderr, "Unknown precision '%s'\n", prec);
    return 1;
}
