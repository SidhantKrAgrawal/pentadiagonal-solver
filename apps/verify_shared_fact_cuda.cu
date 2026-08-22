// verify_shared_fact_cuda.cu — correctness validator for the Shared-Factorisation
// (ADI-structure / shared-coefficient) GPU solver.
//
// Shared-Factorisation's assumption: all systems in a direction share the same
// coefficients.  This tool builds exactly that case, but with coefficients
// that VARY ALONG THE LINE (pseudo-random in i, replicated across systems)
// so that any table-indexing bug is visible — a constant-everywhere fill
// would hide those.  The RHS is random PER SYSTEM.  It then:
//   1. solves with the general-coefficient production path (auto),
//   2. re-uploads the RHS and solves with Shared-Factorisation forced,
//   3. checks both solutions' independent backward residuals, and
//   4. reports the full-grid max|shared-fact - general| (round-off level iff
//      Shared-Factorisation is correct).
// Covers solvedim 0 (x), 1 (y), 2 (z).
//
// Usage: ./verify_shared_fact_cuda [N] [double|float] [solvedim 0|1|2] [n_sample]

#include "pentadsolver.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

template <typename Float>
static void run(int N, const char *prec_name, int solvedim, size_t n_sample) {
    const int NDIMS = 3;
    int dims[3] = {N, N, N};
    size_t n_total = (size_t)N * N * N;
    size_t bytes   = n_total * sizeof(Float);
    size_t stride  = 1;
    for (int k = 0; k < solvedim; k++) { stride *= (size_t)dims[k]; }
    size_t n_lines = n_total / (size_t)N;

    // Per-position (along the line) coefficient profiles, shared by all
    // systems; strictly diagonally dominant; boundary couplings zeroed.
    std::vector<double> pds(N), pdl(N), pd(N), pdu(N), pdw(N);
    std::mt19937 rng(11723u);
    std::uniform_real_distribution<double> off(0.0, 1.0);
    std::uniform_real_distribution<double> diag(5.0, 6.0);
    for (int i = 0; i < N; i++) {
        pds[i] = (i >= 2)     ? off(rng) : 0.0;
        pdl[i] = (i >= 1)     ? off(rng) : 0.0;
        pd[i]  = diag(rng);
        pdu[i] = (i <= N - 2) ? off(rng) : 0.0;
        pdw[i] = (i <= N - 3) ? off(rng) : 0.0;
    }

    std::vector<Float> hds(n_total), hdl(n_total), hd(n_total), hdu(n_total),
        hdw(n_total), hb(n_total);
    std::uniform_real_distribution<double> rhs(-1.0, 1.0);
    for (size_t g = 0; g < n_total; g++) {
        size_t i = (g / stride) % (size_t)N;
        hds[g] = (Float)pds[i];
        hdl[g] = (Float)pdl[i];
        hd[g]  = (Float)pd[i];
        hdu[g] = (Float)pdu[i];
        hdw[g] = (Float)pdw[i];
        hb[g]  = (Float)rhs(rng); // per-system random RHS
    }

    Float *d_ds, *d_dl, *d_d, *d_du, *d_dw, *d_x;
    check(cudaMalloc(&d_ds, bytes), "ds");
    check(cudaMalloc(&d_dl, bytes), "dl");
    check(cudaMalloc(&d_d,  bytes), "d");
    check(cudaMalloc(&d_du, bytes), "du");
    check(cudaMalloc(&d_dw, bytes), "dw");
    check(cudaMalloc(&d_x,  bytes), "x");
    check(cudaMemcpy(d_ds, hds.data(), bytes, cudaMemcpyHostToDevice), "cp ds");
    check(cudaMemcpy(d_dl, hdl.data(), bytes, cudaMemcpyHostToDevice), "cp dl");
    check(cudaMemcpy(d_d,  hd.data(),  bytes, cudaMemcpyHostToDevice), "cp d");
    check(cudaMemcpy(d_du, hdu.data(), bytes, cudaMemcpyHostToDevice), "cp du");
    check(cudaMemcpy(d_dw, hdw.data(), bytes, cudaMemcpyHostToDevice), "cp dw");

    pentadsolver_handle_t handle{};
    size_t buf_bytes = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, solvedim);
    void *d_buf = nullptr;
    if (buf_bytes > 0) check(cudaMalloc(&d_buf, buf_bytes), "buf");

    auto solve = [&](bool use_shared_fact, std::vector<Float> &out) {
        if (use_shared_fact) {
            setenv("PENTA_XALGO", "shared-fact", 1);
            setenv("PENTA_YZALGO", "shared-fact", 1);
        } else {
            unsetenv("PENTA_XALGO");
            unsetenv("PENTA_YZALGO");
        }
        check(cudaMemcpy(d_x, hb.data(), bytes, cudaMemcpyHostToDevice), "cp b");
        pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x,
                                dims, NDIMS, solvedim, d_buf);
        check(cudaDeviceSynchronize(), "solve");
        out.resize(n_total);
        check(cudaMemcpy(out.data(), d_x, bytes, cudaMemcpyDeviceToHost), "cp x");
        unsetenv("PENTA_XALGO");
        unsetenv("PENTA_YZALGO");
    };

    std::vector<Float> x_gen, x_shared;
    solve(false, x_gen);
    solve(true,  x_shared);

    double max_diff = 0.0;
    for (size_t g = 0; g < n_total; g++) {
        max_diff = std::max(max_diff,
                            std::fabs((double)x_shared[g] - (double)x_gen[g]));
    }

    if (n_sample > n_lines) n_sample = n_lines;
    auto resid = [&](const std::vector<Float> &x) {
        double max_r = 0.0, max_b = 0.0;
        for (size_t s = 0; s < n_sample; s++) {
            size_t line  = (n_lines <= n_sample) ? s : s * (n_lines / n_sample);
            size_t outer = line / stride;
            size_t inner = line % stride;
            size_t base  = outer * stride * (size_t)N + inner;
            for (int i = 0; i < N; i++) {
                size_t g = base + (size_t)i * stride;
                double lhs =
                    pds[i] * ((i >= 2)     ? (double)x[g - 2 * stride] : 0.0) +
                    pdl[i] * ((i >= 1)     ? (double)x[g - stride]     : 0.0) +
                    pd[i]  * (double)x[g] +
                    pdu[i] * ((i <= N - 2) ? (double)x[g + stride]     : 0.0) +
                    pdw[i] * ((i <= N - 3) ? (double)x[g + 2 * stride] : 0.0);
                max_r = std::max(max_r, std::fabs(lhs - (double)hb[g]));
                max_b = std::max(max_b, std::fabs((double)hb[g]));
            }
        }
        return max_r / max_b;
    };
    double r_gen = resid(x_gen);
    double r_shared = resid(x_shared);

    double tol = (sizeof(Float) == 8) ? 1e-11 : 1e-4;
    bool pass = (r_gen < tol) && (r_shared < tol);
    printf("[verify_shared_fact] N=%d prec=%s solvedim=%d lines_sampled=%zu\n",
           N, prec_name, solvedim, n_sample);
    printf("  backward resid general = %.3e\n", r_gen);
    printf("  backward resid shared-fact  = %.3e\n", r_shared);
    printf("  max|shared-fact - general|  = %.3e   (full grid)\n", max_diff);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
    cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
    if (d_buf) cudaFree(d_buf);
    if (!pass) exit(1);
}

int main(int argc, char **argv) {
    int N            = (argc > 1) ? atoi(argv[1]) : 256;
    const char *prec = (argc > 2) ? argv[2] : "double";
    int solvedim     = (argc > 3) ? atoi(argv[3]) : 0;
    size_t n_sample  = (argc > 4) ? (size_t)atoll(argv[4]) : 2048;
    if (solvedim < 0 || solvedim > 2) {
        fprintf(stderr, "solvedim must be 0, 1 or 2\n");
        return 1;
    }
    if (std::strcmp(prec, "float") == 0) run<float>(N, "float", solvedim, n_sample);
    else run<double>(N, "double", solvedim, n_sample);
    return 0;
}
