// adi_mpi_cpu.cpp: 3-D ADI pentadiagonal solver, MPI CPU version
//
// Methodology mirror of apps/adi_cpu.cpp (which mirrors adi_cuda.cu), with
// a 1-D MPI domain decomposition along z: mpi_dims = {1, 1, np}.  x- and
// y-direction solves are then rank-local; the z-direction solve exercises
// the distributed (allgather reduced-system) MPI solver path in
// src/cpu/mpi/pentad_cpu.cpp.  Within each rank the library remains
// OpenMP+SIMD parallel (hybrid runs: set OMP_NUM_THREADS > 1).
//
// Timing: MPI_Barrier + MPI_Wtime around each direction; the reported time
// is the MAX across ranks (honest distributed convention).
//
// Correctness self-check: prints a global checksum (Allreduce of sum(x))
// after the timed loop, must match across different np for the same N,
// niters and precision (the MPI test suite upstream has thin coverage, so
// this cross-np checksum is the correctness evidence at application scale).
//
// Usage:  mpirun -np P ./adi_mpi_cpu [N] [niters] [precision]

#include "pentadsolver.hpp"

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

template <typename Float>
static void run_adi(int N, int NITERS, const char *precision_name) {
    const int NWARMUP = 2;
    const int NDIMS   = 3;

    int np = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 1-D decomposition along z (dimension index 2 = outermost)
    int mpi_dims[3] = {1, 1, np};
    int periods[3]  = {0, 0, 0};
    MPI_Comm cart_comm{};
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, mpi_dims, periods, 0, &cart_comm);

    pentadsolver_handle_t handle{};
    pentadsolver_create(&handle, &cart_comm, NDIMS, mpi_dims);

    // Local slab: N x N x nz_local (last rank absorbs the remainder)
    int nz_base  = N / np;
    int nz_local = (rank == np - 1) ? N - nz_base * (np - 1) : nz_base;
    int local_dims[3] = {N, N, nz_local};
    size_t n_local = (size_t)N * N * nz_local;

    if (rank == 0) {
        printf("=========================================================\n");
        printf("Pentadiagonal ADI, MPI CPU  (z-slab decomposition)\n");
        printf("Grid: %d x %d x %d   ranks: %d   threads/rank: %d\n",
               N, N, N, np, omp_get_max_threads());
        printf("Precision: %-6s  Warmup: %d iter   Timed: %d iter\n",
               precision_name, NWARMUP, NITERS);
        printf("=========================================================\n\n");
    }

    std::vector<Float> ds(n_local, Float(1.0));
    std::vector<Float> dl(n_local, Float(1.0));
    std::vector<Float> d (n_local, Float(5.0));
    std::vector<Float> du(n_local, Float(1.0));
    std::vector<Float> dw(n_local, Float(1.0));
    std::vector<Float> x (n_local, Float(1.0));

    // Global boundary rows: the z-solve couples across ranks, but the global
    // first/last two planes must carry zeroed sub/super coefficients exactly
    // as in the single-node app (the constant fill already provides interior
    // coupling; the MPI solver takes care of inter-rank coupling).  For x/y
    // the solver zeroes per its internal convention, the constant-fill
    // matrix is diagonally dominant (|5| > 4) so the solve is well-posed.

    size_t buf_bytes = 0;
    for (int dim = 0; dim < 3; dim++) {
        size_t b = pentadsolver_gpsv_batch_buffer_size_ext(
            handle, ds.data(), dl.data(), d.data(), du.data(), dw.data(),
            x.data(), local_dims, NDIMS, dim);
        if (b > buf_bytes) buf_bytes = b;
    }
    std::vector<char> buffer(buf_bytes, 0);
    if (rank == 0) {
        printf("Scratch buffer: %.1f MB per rank\n\n",
               (double)buf_bytes / (1024.0 * 1024.0));
    }

    for (int it = 0; it < NWARMUP; it++) {
        for (int dim = 0; dim < 3; dim++) {
            pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                    du.data(), dw.data(), x.data(),
                                    local_dims, NDIMS, dim, buffer.data());
        }
    }

    double t_dir[3] = {0.0, 0.0, 0.0};
    for (int it = 0; it < NITERS; it++) {
        for (int dim = 0; dim < 3; dim++) {
            MPI_Barrier(cart_comm);
            double t0 = MPI_Wtime();
            pentadsolver_gpsv_batch(handle, ds.data(), dl.data(), d.data(),
                                    du.data(), dw.data(), x.data(),
                                    local_dims, NDIMS, dim, buffer.data());
            MPI_Barrier(cart_comm);
            t_dir[dim] += (MPI_Wtime() - t0) * 1e3;   // ms
        }
    }

    // Max across ranks (they should agree because of the barriers)
    double t_max[3];
    MPI_Reduce(t_dir, t_max, 3, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    // Global checksum for cross-np correctness comparison
    double local_sum = 0.0;
    for (size_t i = 0; i < n_local; i++) local_sum += (double)x[i];
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        double ax = t_max[0] / NITERS, ay = t_max[1] / NITERS,
               az = t_max[2] / NITERS;
        printf("MPI CPU timings (avg over %d iters, barrier+MPI_Wtime, max over ranks):\n",
               NITERS);
        printf("  %-12s %8.3f ms\n", "pentad_x:", ax);
        printf("  %-12s %8.3f ms\n", "pentad_y:", ay);
        printf("  %-12s %8.3f ms   (distributed direction)\n", "pentad_z:", az);
        printf("  %-12s %8.3f ms\n", "total/iter:", ax + ay + az);
        printf("\nGlobal checksum sum(x) = %.12e   (compare across np)\n",
               global_sum);
    }

    pentadsolver_destroy(&handle);
    MPI_Comm_free(&cart_comm);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const int   N      = (argc > 1) ? atoi(argv[1]) : 256;
    const int   NITERS = (argc > 2) ? atoi(argv[2]) : 10;
    const char *prec   = (argc > 3) ? argv[3] : "double";

    int ret = 0;
    if (std::strcmp(prec, "float") == 0) {
        run_adi<float>(N, NITERS, "float");
    } else if (std::strcmp(prec, "double") == 0) {
        run_adi<double>(N, NITERS, "double");
    } else {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            fprintf(stderr, "Unknown precision '%s' (expected 'double' or 'float')\n", prec);
        ret = 1;
    }

    MPI_Finalize();
    return ret;
}
