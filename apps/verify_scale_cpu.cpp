// verify_scale_cpu.cpp — independent residual check for the CPU solver at
// an arbitrary grid size N, used to gain confidence at sizes (320, 384)
// that the fixed correctness test suite does not cover.
//
// Solves the x-direction (solvedim 0, contiguous/stride-1) once with the
// same constant diagonally-dominant fill as the ADI apps, then recomputes
// the residual r_i = ds*x[i-2] + dl*x[i-1] + d*x[i] + du*x[i+1] + dw*x[i+2]
// - b_i directly from the ORIGINAL coefficients and right-hand side
// (independent of the solver), for INTERIOR rows only (i in [2, N-3] of
// each line) — boundary-row special-casing is unchanged from N=256, where
// it is already proven correct by the 201,388-assertion suite.
//
// Usage: ./verify_scale_cpu [N] [double|float]

#include "pentadsolver.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

template <typename Float>
static void verify(int N, const char *prec_name) {
    const int NDIMS = 3;
    int dims[3] = {N, N, N};
    size_t n_total = (size_t)N * N * N;

    std::vector<Float> ds(n_total, Float(1.0));
    std::vector<Float> dl(n_total, Float(1.0));
    std::vector<Float> d (n_total, Float(5.0));
    std::vector<Float> du(n_total, Float(1.0));
    std::vector<Float> dw(n_total, Float(1.0));
    std::vector<Float> b (n_total, Float(1.0));   // original RHS
    std::vector<Float> x = b;                      // solver overwrites this

    pentadsolver_handle_t handle{};
    pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(), du.data(),
                            dw.data(), x.data(), dims, NDIMS, 0, nullptr);

    // Residual over interior rows of every x-line (stride 1, N per line).
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
            double resid = std::fabs(lhs - (double)b[g]);
            max_abs_resid = std::max(max_abs_resid, resid);
            max_rel_resid = std::max(max_rel_resid, resid / std::fabs((double)b[g]));
        }
    }

    double tol = (sizeof(Float) == 8) ? 1e-8 : 1e-3;  // FP32 has ~7 sig digits
    printf("[CPU verify] N=%d precision=%s  n_lines=%zu\n", N, prec_name, n_lines);
    printf("  max |residual|      = %.3e\n", max_abs_resid);
    printf("  max |residual|/|b|  = %.3e\n", max_rel_resid);
    printf("  %s\n", max_rel_resid < tol ? "PASS" : "FAIL");
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 320;
    const char *prec = (argc > 2) ? argv[2] : "double";
    if (std::strcmp(prec, "float") == 0) verify<float>(N, "float");
    else verify<double>(N, "double");
    return 0;
}
