#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
#include "preproc.hpp"
#include "pentadsolver.hpp"
#include <cmath>
#include <iomanip>

//
// linux timing routine
//
#include <sys/time.h>

inline double elapsed_time(double *et) {
  struct timeval t;
  double old_time = *et;

  gettimeofday( &t, (struct timezone *)0 );
  *et = t.tv_sec + t.tv_usec*1.0e-6;

  return *et - old_time;
}

inline void timing_start(int prof, double *timer) {
  if(prof==1) elapsed_time(timer);
}

inline void timing_end(int prof, double *timer, double *elapsed_accumulate, char *str) {
  double elapsed;
  if(prof==1) {
    elapsed = elapsed_time(timer);
    *elapsed_accumulate += elapsed;
  }
}


 int main(int argc, char* argv[]) {
  double timer, timer2, elapsed, elapsed_total, elapsed_preproc, elapsed_u_momentum, elapsed_v_momentum, elapsed_velocity_correction, elapsed_u,
  elapsed_prime_velocity, elapsed_pressure_correction, elapsed_pentad_x, elapsed_pentad_y;

  int i, j, ind, it;
  int nx, nx_pad, ny, iter, max_iter, opt, prof;
  int N_MAX = 1024;
  // 'h_' prefix - CPU (host) memory space
  FP  *__restrict__ h_y, *__restrict__ h_tmp, *__restrict__ h_x,
      *__restrict__ h_sx, *__restrict__ h_lx, *__restrict__ h_dx, *__restrict__ h_ux, *__restrict__ h_wx,
      *__restrict__ h_sy, *__restrict__ h_ly, *__restrict__ h_dy, *__restrict__ h_uy, *__restrict__ h_wy,
      *__restrict__ tmp , *__restrict__ h_p,
      err, lambda=1.0f; // lam = dt/dx^2

  // Set defaults options
  nx   = 16;
  ny   = 16;
  max_iter = 10;
  iter = 10;
  opt  = 0;
  prof = 1;

  if( nx>N_MAX || ny>N_MAX) {
    printf("Dimension can not exceed N_MAX=%d due to hard-coded local array sizes\n", N_MAX);
    exit(1);
  }

  // allocate memory for arrays
  nx_pad = (1+((nx-1)/SIMD_VEC))*SIMD_VEC; // Compute padding for vecotrization
  h_y  = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_tmp= (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_x = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_p = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_sx = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_lx = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_dx = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_ux = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_wx = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_sy = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_ly = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_dy = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_uy = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);
  h_wy = (double *)_mm_malloc(sizeof(double)*nx_pad*ny,SIMD_WIDTH);

  printf("\nGrid dimensions: %d x %d\n", nx, ny);
  printf("Check parameters: SIMD_WIDTH = %d, sizeof(FP) = %d, nx_pad = %d , SIMD_VEC =%d \n",SIMD_WIDTH,sizeof(FP),nx_pad,SIMD_VEC);


    //Initialize

    for(j=0; j<ny; j++) {
      for(i=0; i<nx; i++) {
        ind =j*nx_pad + i;
        h_sx[ind] = 1.0;
        h_sy[ind] = 1.0;
        h_lx[ind] = 1.0;
        h_ly[ind] = 1.0;
        h_dx[ind] = 1.0;
        h_dy[ind] = 1.0;
        h_ux[ind] = 1.0;
        h_uy[ind] = 1.0;
        h_wx[ind] = 1.0;
        h_wy[ind] = 1.0;
        h_x[ind]  = 0.0;
      }
  }
  // reset elapsed time counters
  elapsed_total      = 0.0;
  elapsed_u_momentum = 0.0;
  elapsed_v_momentum = 0.0;
  elapsed_preproc = 0.0;
  elapsed_prime_velocity = 0.0;
  elapsed_velocity_correction = 0.0;
  elapsed_pressure_correction = 0.0;
  elapsed_pentad_x  = 0.0;
  elapsed_pentad_y  = 0.0;
  elapsed_u         = 0.0;

   //problem domain - 2D Cavity flow using SIMPLE algorithm
  //grid points along x and y dimension is 256 by default
   double dx = 1.0/nx;
   double dy = 1.0/ny;
   const double Re = 100;                  //Reynold's Number
   const double pressure_const = 0.2;
   const double vel_const = 0.7;         //velocity under relaxation
   const double epsilonU = 1e-3;       
   const double epsilonP = 1e-4; 
   double  res_p, res_u, res_v  = 0;
   double error = 0.0;
  //u,v,p variable declaration
  std::vector<std::vector<double>> p(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> p_star(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> p_prime(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> p_rhs(nx, std::vector<double>(ny, 0.0)); 
  std::vector<std::vector<double>> p_res(ny+2, std::vector<double>(nx+2, 0.0)); 

  std::vector<std::vector<double>> u(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> u_star(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> u_prime(ny+2, std::vector<double>(nx+1, 0.0)); 
  std::vector<std::vector<double>> u_prev(ny+2, std::vector<double>(nx+2, 0.0)); 
  
  std::vector<std::vector<double>> v(ny+2, std::vector<double>(nx+2, 0.0)); 
  std::vector<std::vector<double>> v_star(ny+2, std::vector<double>(nx+2, 0.0));
  std::vector<std::vector<double>> v_prime(ny+1, std::vector<double>(nx+2, 0.0));
  std::vector<std::vector<double>> v_prev(nx, std::vector<double>(ny+1, 0.0));
  std::vector<std::vector<double>> v_res(nx, std::vector<double>(ny+1, 0.0));

  //-------Boundary coefficients---------
  std::vector<std::vector<double>> A_p(ny+2, std::vector<double>(nx+2, 1.0));
  std::vector<std::vector<double>> A_e(ny+2, std::vector<double>(nx+2, 1.0));
  std::vector<std::vector<double>> A_w(ny+2, std::vector<double>(nx+2, 1.0));
  std::vector<std::vector<double>> A_n(ny+2, std::vector<double>(nx+2, 1.0));
  std::vector<std::vector<double>> A_s(ny+2, std::vector<double>(nx+2, 1.0));

  std::vector<std::vector<double>> x_vec(ny+2, std::vector<double>(nx+2, 0.0));
  std::vector<std::vector<double>> y_vec(ny+2, std::vector<double>(nx+2, 0.0));
  //Boundary condition 
  //Top wall velocity is 1m/s, velocity at the rest of the walls is 0
  for(int i =1; i<nx; i++){
    u[0][i] = 1.0;
    u_star[0][i] = 1.0;
    u_prime[0][i] = 1.0;
  }
    u_prime[0][nx] = 0.0;

  elapsed_time(&timer2);
  for(it = 0; it<iter; it++) {
    

    //calculate u-momentum
    timing_start(prof, &timer);
    uv_momentum<double>(nx_pad, ny, dx, dy, Re, p, vel_const, u_star, v_star, u_prime, v_prime,
                    A_e,A_w,A_n,A_s,A_p,x_vec,y_vec);
    timing_end(prof, &timer, &elapsed_u_momentum, "uv_matrix");

    timing_start(prof, &timer);
    solve(nx, ny, u, u_star,A_p, A_e,A_w,A_n,A_s,x_vec,vel_const,epsilonU,res_u);
    timing_end(prof, &timer, &elapsed_u, "u_momentum");


    solve(nx, ny, v, v_star,A_p, A_e,A_w,A_n,A_s,y_vec,vel_const,epsilonU,res_v);
     
    // Pentadiagonal solver option arguemnt's setup
    int ndim = 2;  // Number of dimensions of the (hyper)cubic data structure.
    int dims[2];   // Array containing the sizes of each ndim dimensions. size(dims) == ndim <=MAXDIM
    //int pads[2];   // Padded sizes along each ndim number of dimensions
    dims[0] = nx;
    dims[1] = ny;

    //calculate uv_prime velocity
    timing_start(prof, &timer);
      uv_prime<double>(nx, ny, dx, dy, vel_const, u, v, u_prime,v_prime, p, A_p, iter);
    timing_end(prof, &timer, &elapsed_v_momentum, "uv_prime");
         
    //
    // calculate r.h.s. and set penta-diagonal coefficients
    //

    timing_start(prof, &timer);
      pressure_coef_matrix<double>(nx, ny, nx_pad, vel_const, dx, dy, u_prime, v_prime, A_p,
                          h_x, h_sx, h_lx, h_dx, h_ux, h_wx, h_sy, h_ly, h_dy, h_uy, h_wy);
    timing_end(prof, &timer, &elapsed_preproc, "pressure_coefficient");

    //----------------solve pressure gradient using pentadsolver-----------------
    //
    // perform penta-diagonal solves in x-direction
    //

    timing_start(prof, &timer);

    int solvedim = 0;   // user chosen dimension for which the solution is performed
    #if FPPREC == 0
        pentadsolver_gpsv_batch(nullptr, h_sx, h_lx, h_dx, h_ux, h_wx, h_x, dims, ndim, solvedim, nullptr);
    #elif FPPREC == 1
       pentadsolver_gpsv_batch(nullptr, h_sx, h_lx, h_dx, h_ux, h_wx, h_x, dims, ndim, solvedim, nullptr);
    #endif
      timing_end(prof, &timer, &elapsed_pentad_x, "pentad_x for pressure equation");
 
    //
    // perform penta-diagonal solves in y-direction
    //
    timing_start(prof, &timer);

    solvedim = 1;   // user chosen dimension for which the solution is performed
    #if FPPREC == 0
      pentadsolver_gpsv_batch(nullptr, h_sy, h_ly, h_dy, h_uy, h_wy, h_x, dims, ndim, solvedim, nullptr);
    #elif FPPREC == 1
      pentadsolver_gpsv_batch(nullptr, h_sy, h_ly, h_dy, h_uy, h_wy, h_x, dims, ndim, solvedim, nullptr);
    #endif

    timing_end(prof, &timer, &elapsed_pentad_y, "pentad_y for pressure equation");

     update_values<double>(nx,ny,h_x, p_prime);
    
    //calculate pressure_correction
    timing_start(prof, &timer);
      pressure_correction<double>(nx, ny, pressure_const, p, p_star, p_prime);
    timing_end(prof, &timer, &elapsed_pressure_correction, "pressure correction");
    

    //calculate velocity_correction
    timing_start(prof, &timer);
      velocity_correction<FP>(nx, ny, vel_const, dx, dy, u, v, u_star, v_star, p_prime,A_p);
    timing_end(prof, &timer, &elapsed_velocity_correction, "velocity correction");


   //calculate updated prime values for velocities
    timing_start(prof, &timer);
      update_velocities<FP>(nx, ny, vel_const, dx, dy, u_prime, v_prime, p_prime,A_p);
    timing_end(prof, &timer, &elapsed_prime_velocity, "updated prime velocities");
    
      double rms_u, rms_v, rms_p;  
      rms_p = 0;
      rms_u = std::sqrt(std::abs(res_u));
      rms_v = std::sqrt(std::abs(res_v));   
     
  }

  elapsed = elapsed_time(&timer2);
  elapsed_total = elapsed;
  printf("\nADI total execution time for %d iterations (sec): %f (s) \n", iter, elapsed);
  fflush(0);

  int ldim=nx_pad;
 
  _mm_free(h_y);
  _mm_free(h_sx);
  _mm_free(h_lx);
  _mm_free(h_dx);
  _mm_free(h_ux);
  _mm_free(h_wx);
  _mm_free(h_sy);
  _mm_free(h_ly);
  _mm_free(h_dy);
  _mm_free(h_uy);
  _mm_free(h_wy);
  _mm_free(h_x);

  printf("Done.\n");

  // Print execution times
  if(prof == 0) {
    printf("Avg(per iter) \n[total]\n");
    printf("%f\n", elapsed_total/iter);
  }
  else if(prof == 1) {

  printf("Time per element averaged on %d iterations: \n[total] \n[uv_matrix] \t[u_momentum_solve] \t[uv_prime] \t[vel_correction] \t[prime_vel_correction]\n", iter);
  printf("%e \t%e \t%e \t%e \t%e \t%e\n",
      (elapsed_total/iter)/(nx*ny),
      (elapsed_u_momentum/iter)/(nx*ny),
      (elapsed_u/iter)/(nx*ny),
      (elapsed_v_momentum/iter)/(nx*ny),
      (elapsed_velocity_correction/iter)/(nx*ny),
      (elapsed_prime_velocity/iter)/(nx*ny));
  
  printf("\nTime per element averaged on %d iterations: \n[total] \t[prepro] \t[pentad_x] \t[pentad_y]  \t[pressure_correction]\n", iter);
  printf("%e \t%e \t%e \t%e \t%e\n",
      (elapsed_total),
      (elapsed_preproc),
      (elapsed_pentad_x),
      (elapsed_pentad_y),
      (elapsed_pressure_correction));
  }
 
  exit(0);

}
