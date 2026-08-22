// fp32_accuracy_cuda.cu — FP32-vs-FP64 accuracy study for the GPU pentadiagonal
// x-solver.
//
// The residual-only verify tools (verify_scale_*) measure BACKWARD error
// (how well the returned x satisfies A x = b).  This tool additionally
// measures FORWARD error: the deviation of the FP32 solution from an
// INDEPENDENT double-precision reference, so we can quantify "how much
// accuracy does the FP32 speedup cost?" and whether that error grows with the
// problem size N or with the algorithm choice (thomas-pcr POP-PCR's redundant
// arithmetic vs plain Thomas).
//
// Method:
//   * Build a reproducible (seeded) RANDOM, strictly diagonally-dominant
//     pentadiagonal system.  Each x-line is a genuine, independent N x N
//     system (out-of-range boundary couplings are zeroed), so every row's
//     residual (including the boundary rows) is meaningful.
//   * The SAME coefficients (as stored in the chosen Float type) drive both
//     the GPU solve and a host serial double-precision Thomas reference — so
//     the only difference measured is round-off, not a different problem.
//   * Report, over a sample of lines:
//       backward: max |A x - b| / |b|
//       forward : max |x_gpu - x_ref| (abs) and / max|x_ref| (rel), plus RMS.
//
// Precision + algorithm are selected exactly like the rest of the campaign:
//   PENTA_XALGO in {naive,transpose,thomas-pcr,shared-fact,auto} forces the kernel.
// Usage: ./fp32_accuracy_cuda [N] [double|float] [n_sample_lines]

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

// Independent host serial double-precision pentadiagonal Thomas solve of ONE
// line of length N.  Coefficients come in as double (already cast from the
// Float values the GPU actually used).  Returns the solution in xr.
static void host_thomas_penta(int N, const double *ds, const double *dl,
                              const double *d, const double *du,
                              const double *dw, const double *b,
                              std::vector<double> &xr,
                              std::vector<double> &du2,
                              std::vector<double> &dw2,
                              std::vector<double> &x2) {
    for (int i = 0; i < N; i++) {
        double du2_2 = (i >= 2) ? du2[i - 2] : 0.0;
        double dw2_2 = (i >= 2) ? dw2[i - 2] : 0.0;
        double du2_1 = (i >= 1) ? du2[i - 1] : 0.0;
        double dw2_1 = (i >= 1) ? dw2[i - 1] : 0.0;
        double x2_2  = (i >= 2) ? x2[i - 2]  : 0.0;
        double x2_1  = (i >= 1) ? x2[i - 1]  : 0.0;

        double ddl = dl[i] - ds[i] * du2_2;
        double dd  = d[i]  - ds[i] * dw2_2 - ddl * du2_1;
        du2[i] = (du[i] - ddl * dw2_1) / dd;
        dw2[i] = dw[i] / dd;
        x2[i]  = (b[i] - ds[i] * x2_2 - ddl * x2_1) / dd;
    }
    xr[N - 1] = x2[N - 1];
    if (N >= 2) xr[N - 2] = x2[N - 2] - du2[N - 2] * xr[N - 1];
    for (int i = N - 3; i >= 0; i--)
        xr[i] = x2[i] - du2[i] * xr[i + 1] - dw2[i] * xr[i + 2];
}

template <typename Float>
static void run(int N, const char *prec_name, size_t n_sample) {
    const int NDIMS = 3;
    int dims[3] = {N, N, N};
    size_t n_total = (size_t)N * N * N;
    size_t bytes   = n_total * sizeof(Float);
    size_t n_lines = n_total / (size_t)N;

    std::vector<Float> hds(n_total), hdl(n_total), hd(n_total), hdu(n_total),
        hdw(n_total), hx(n_total);

    // Reproducible random strictly-diagonally-dominant system.
    // off-diagonals in [0,1); diagonal = 5 + [0,1) so |d| > sum|offdiag| (<=4).
    std::mt19937 rng(12345u);
    std::uniform_real_distribution<double> off(0.0, 1.0);
    std::uniform_real_distribution<double> diag(5.0, 6.0);
    std::uniform_real_distribution<double> rhs(-1.0, 1.0);
    for (size_t line = 0; line < n_lines; line++) {
        size_t base = line * (size_t)N;
        for (int i = 0; i < N; i++) {
            size_t g = base + (size_t)i;
            // zero the couplings that reach outside this line's N x N system
            hds[g] = (Float)((i >= 2)     ? off(rng) : 0.0);
            hdl[g] = (Float)((i >= 1)     ? off(rng) : 0.0);
            hd[g]  = (Float)diag(rng);
            hdu[g] = (Float)((i <= N - 2) ? off(rng) : 0.0);
            hdw[g] = (Float)((i <= N - 3) ? off(rng) : 0.0);
            hx[g]  = (Float)rhs(rng);
        }
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
    check(cudaMemcpy(d_x,  hx.data(),  bytes, cudaMemcpyHostToDevice), "cp x");

    pentadsolver_handle_t handle{};
    size_t buf_bytes = pentadsolver_gpsv_batch_buffer_size_ext(
        handle, d_ds, d_dl, d_d, d_du, d_dw, d_x, dims, NDIMS, 0);
    void *d_buf = nullptr;
    if (buf_bytes > 0) check(cudaMalloc(&d_buf, buf_bytes), "buf");

    pentadsolver_gpsv_batch(handle, d_ds, d_dl, d_d, d_du, d_dw, d_x,
                            dims, NDIMS, 0, d_buf);
    check(cudaDeviceSynchronize(), "solve");

    std::vector<Float> xg(n_total);
    check(cudaMemcpy(xg.data(), d_x, bytes, cudaMemcpyDeviceToHost), "cp x back");

    // Compare a sample of lines against the independent double reference.
    if (n_sample > n_lines) n_sample = n_lines;
    std::vector<double> ds_d(N), dl_d(N), d_d_(N), du_d(N), dw_d(N), b_d(N);
    std::vector<double> xr(N), du2(N), dw2(N), x2(N);

    double max_abs_fwd = 0.0, sum_sq_fwd = 0.0, max_ref = 0.0;
    double max_abs_bwd = 0.0, max_abs_b = 0.0;  // norm-wise (inf-norm) backward error
    size_t counted = 0;
    for (size_t s = 0; s < n_sample; s++) {
        // spread the sample across the whole batch
        size_t line = (n_lines <= n_sample) ? s : (s * (n_lines / n_sample));
        size_t base = line * (size_t)N;
        for (int i = 0; i < N; i++) {
            size_t g = base + (size_t)i;
            ds_d[i] = (double)hds[g]; dl_d[i] = (double)hdl[g];
            d_d_[i] = (double)hd[g];  du_d[i] = (double)hdu[g];
            dw_d[i] = (double)hdw[g]; b_d[i]  = (double)hx[g];
        }
        host_thomas_penta(N, ds_d.data(), dl_d.data(), d_d_.data(),
                          du_d.data(), dw_d.data(), b_d.data(), xr, du2, dw2, x2);
        for (int i = 0; i < N; i++) {
            size_t g = base + (size_t)i;
            double fwd = std::fabs((double)xg[g] - xr[i]);
            max_abs_fwd = std::max(max_abs_fwd, fwd);
            sum_sq_fwd += fwd * fwd;
            max_ref = std::max(max_ref, std::fabs(xr[i]));
            // backward residual on this line (all rows; boundaries zeroed)
            double lhs = ds_d[i] * ((i >= 2) ? (double)xg[base + i - 2] : 0.0)
                       + dl_d[i] * ((i >= 1) ? (double)xg[base + i - 1] : 0.0)
                       + d_d_[i] * (double)xg[g]
                       + du_d[i] * ((i <= N - 2) ? (double)xg[base + i + 1] : 0.0)
                       + dw_d[i] * ((i <= N - 3) ? (double)xg[base + i + 2] : 0.0);
            max_abs_bwd = std::max(max_abs_bwd, std::fabs(lhs - b_d[i]));
            max_abs_b   = std::max(max_abs_b, std::fabs(b_d[i]));
            counted++;
        }
    }
    double rms_fwd = std::sqrt(sum_sq_fwd / (double)counted);
    double rel_fwd = max_abs_fwd / std::max(1e-30, max_ref);
    double rel_bwd = max_abs_bwd / std::max(1e-30, max_abs_b);

    const char *algo = std::getenv("PENTA_XALGO");
    printf("[FP32 accuracy] N=%d  precision=%-6s  algo=%-6s  lines_sampled=%zu (rows=%zu)\n",
           N, prec_name, algo ? algo : "auto", n_sample, counted);
    printf("  backward: ||Ax-b||inf/||b||inf = %.3e   (max|Ax-b|=%.3e)\n",
           rel_bwd, max_abs_bwd);
    printf("  forward : max|x_gpu - x_ref| = %.3e   (rel to max|x_ref|=%.3e -> %.3e)\n",
           max_abs_fwd, max_ref, rel_fwd);
    printf("  forward : RMS|x_gpu - x_ref| = %.3e\n", rms_fwd);

    cudaFree(d_ds); cudaFree(d_dl); cudaFree(d_d);
    cudaFree(d_du); cudaFree(d_dw); cudaFree(d_x);
    if (d_buf) cudaFree(d_buf);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 256;
    const char *prec = (argc > 2) ? argv[2] : "float";
    size_t n_sample = (argc > 3) ? (size_t)atoll(argv[3]) : 2048;
    if (std::strcmp(prec, "float") == 0) run<float>(N, "float", n_sample);
    else run<double>(N, "double", n_sample);
    return 0;
}
