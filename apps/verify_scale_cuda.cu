// verify_scale_cuda.cu — independent residual check for the GPU solver at
// an arbitrary grid size N (see verify_scale_cpu.cpp for full rationale).
// Solves x-direction once (through whatever PENTA_XALGO / production
// dispatch selects), copies x back to host, and recomputes the residual
// from the original coefficients — independent of the solver's own
// internal correctness paths.
//
// Usage: ./verify_scale_cuda [N] [double|float]

#include "pentadsolver.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
static void verify(int N, const char *prec_name) {
    const int NDIMS = 3;
    int dims[3] = {N, N, N};
    size_t n_total = (size_t)N * N * N;
    size_t bytes   = n_total * sizeof(Float);

    Float *d_ds, *d_dl, *d_d, *d_du, *d_dw, *d_x;
    check(cudaMalloc(&d_ds, bytes), "ds");
    check(cudaMalloc(&d_dl, bytes), "dl");
    check(cudaMalloc(&d_d,  bytes), "d");
    check(cudaMalloc(&d_du, bytes), "du");
    check(cudaMalloc(&d_dw, bytes), "dw");
    check(cudaMalloc(&d_x,  bytes), "x");

    unsigned blocks = (unsigned)((n_total + 255) / 256);
    fill_kernel<<<blocks, 256>>>(d_ds, Float(1.0), n_total);
    fill_kernel<<<blocks, 256>>>(d_dl, Float(1.0), n_total);
    fill_kernel<<<blocks, 256>>>(d_d,  Float(5.0), n_total);
    fill_kernel<<<blocks, 256>>>(d_du, Float(1.0), n_total);
    fill_kernel<<<blocks, 256>>>(d_dw, Float(1.0), n_total);
    fill_kernel<<<blocks, 256>>>(d_x,  Float(1.0), n_total);
    check(cudaDeviceSynchronize(), "init");

    pentadsolver_handle_t handle{};
    size_t buf_bytes = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, 0);
    void *d_buf = nullptr;
    if (buf_bytes > 0) check(cudaMalloc(&d_buf, buf_bytes), "buf");

    pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x,
                            dims, NDIMS, 0, d_buf);
    check(cudaDeviceSynchronize(), "solve");

    std::vector<Float> ds(n_total), dl(n_total), d(n_total), du(n_total),
        dw(n_total), x(n_total);
    check(cudaMemcpy(ds.data(), d_ds, bytes, cudaMemcpyDeviceToHost), "cp ds");
    check(cudaMemcpy(dl.data(), d_dl, bytes, cudaMemcpyDeviceToHost), "cp dl");
    check(cudaMemcpy(d.data(),  d_d,  bytes, cudaMemcpyDeviceToHost), "cp d");
    check(cudaMemcpy(du.data(), d_du, bytes, cudaMemcpyDeviceToHost), "cp du");
    check(cudaMemcpy(dw.data(), d_dw, bytes, cudaMemcpyDeviceToHost), "cp dw");
    check(cudaMemcpy(x.data(),  d_x,  bytes, cudaMemcpyDeviceToHost), "cp x");

    // Original RHS was the constant fill value 1.0 everywhere.
    const double b_val = 1.0;
    double max_abs_resid = 0.0, max_rel_resid = 0.0;
    size_t n_lines = n_total / N;
    for (size_t line = 0; line < n_lines; line++) {
        size_t base = line * (size_t)N;
        for (int i = 2; i < N - 2; i++) {
            size_t g = base + i;
            double lhs = (double)ds[g] * (double)x[g - 2]
                       + (double)dl[g] * (double)x[g - 1]
                       + (double)d[g]  * (double)x[g]
                       + (double)du[g] * (double)x[g + 1]
                       + (double)dw[g] * (double)x[g + 2];
            double resid = std::fabs(lhs - b_val);
            max_abs_resid = std::max(max_abs_resid, resid);
            max_rel_resid = std::max(max_rel_resid, resid / std::fabs(b_val));
        }
    }

    double tol = (sizeof(Float) == 8) ? 1e-8 : 1e-3;
    printf("[GPU verify] N=%d precision=%s  n_lines=%zu\n", N, prec_name, n_lines);
    printf("  max |residual|      = %.3e\n", max_abs_resid);
    printf("  max |residual|/|b|  = %.3e\n", max_rel_resid);
    printf("  %s\n", max_rel_resid < tol ? "PASS" : "FAIL");

    cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
    cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
    if (d_buf) cudaFree(d_buf);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 320;
    const char *prec = (argc > 2) ? argv[2] : "double";
    if (std::strcmp(prec, "float") == 0) verify<float>(N, "float");
    else verify<double>(N, "double");
    return 0;
}
