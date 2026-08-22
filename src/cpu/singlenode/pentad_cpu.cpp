#include  <transpose.hpp>
#include "pentad_simd.hpp"
#include "pentad_common.hpp"
#include "pentadsolver.hpp"
#include "pentadsolver_handle.hpp"
#include "timing.h"
#include <cassert>
#include <functional>              // for multiplies
#include <numeric>                 // for accumulate
#define ROUND_DOWN(N, step) (((N) / (step)) * step)
constexpr size_t N_MAX = 1024; // FIXME move to parameter, define

inline void load(SIMD_REG *__restrict__ dst, const FP *__restrict__ src, int n,
                 int pad) {
  for (int i = 0; i < SIMD_VEC; i++) {
    dst[i] = SIMD_LOAD_P(&src[i * pad + n]);
  }
}

inline void store(FP *__restrict__ dst, SIMD_REG *__restrict__ src, int n,
                  int pad) {
  for (int i = 0; i < SIMD_VEC; i++) {
    SIMD_STORE_P(&dst[i * pad + n], src[i]);
  }
}

#if __AVX__
#  if FPPREC == 0
#    define LOAD(reg, array,n, N)                                             \
      load(reg, array, n,N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose4x4_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose4x4_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#endif

template <typename REAL>
void pentad_x_transpose(const REAL *__restrict s, const REAL *__restrict l,
                      const REAL *__restrict d, const REAL *__restrict u, const REAL *__restrict w,
                      REAL *__restrict x, int sys_size, int sys_pad) {
  assert(sys_pad % SIMD_VEC == 0);

  SIMD_REG ss;
  SIMD_REG ll;
  SIMD_REG dd;
  SIMD_REG uu;
  SIMD_REG uu1;
  SIMD_REG ww;
  SIMD_REG ww1;
  SIMD_REG xx;
  SIMD_REG xx1;
  SIMD_REG temp_d;
  SIMD_REG temp_x;

  SIMD_REG s_reg[SIMD_VEC];
  SIMD_REG l_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  SIMD_REG u_reg[SIMD_VEC];
  SIMD_REG w_reg[SIMD_VEC];
  SIMD_REG x_reg[SIMD_VEC];

  SIMD_REG u2[N_MAX];
  SIMD_REG w2[N_MAX];
  SIMD_REG x2[N_MAX];

  //
  // forward pass
  //
  LOAD(s_reg, s, 0, sys_pad);
  LOAD(l_reg, l, 0, sys_pad);
  LOAD(d_reg, d, 0, sys_pad);
  LOAD(u_reg, u, 0, sys_pad);
  LOAD(w_reg, w, 0, sys_pad);
  LOAD(x_reg, x, 0, sys_pad);
  
  //row 0
  dd    = d_reg[0];
  uu    = u_reg[0];
  uu    = SIMD_DIV_P(uu, dd);
  ww    = w_reg[0];
  ww    = SIMD_DIV_P(ww, dd);
  xx    = x_reg[0];
  xx    = SIMD_DIV_P(xx, dd);
  u2[0] = uu;
  w2[0] = ww;
  x2[0] = xx;

  //row 1
  ll    = l_reg[1];
  dd    = d_reg[1];
  dd    = SIMD_SUB_P(dd, SIMD_MUL_P(ll, uu));
  uu    = u_reg[1];
  uu    = SIMD_SUB_P(uu, SIMD_MUL_P(ll, ww));
  uu    = SIMD_DIV_P(uu,dd);
  ww    = w_reg[1];
  ww    = SIMD_DIV_P(ww,dd);
  xx1   = xx;
  xx    = x_reg[1];
  xx    = SIMD_SUB_P(xx, SIMD_MUL_P(ll, xx1));
  xx    = SIMD_DIV_P(xx,dd);
  u2[1] = uu;
  w2[1] = ww;
  x2[1] = xx;

  for (int i = 2; i < SIMD_VEC; i++) {
    ss      = s_reg[i];
    xx1     = x2[i-2];
    uu1     = u2[i-2];
    ww1     = w2[i-2];
    ll      = SIMD_SUB_P(l_reg[i], SIMD_MUL_P(ss, uu1));
    temp_d  = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(ss, ww1));
    dd      = SIMD_SUB_P(temp_d, SIMD_MUL_P(ll, uu));
      uu      = SIMD_SUB_P(u_reg[i], SIMD_MUL_P(ll, ww));
    uu      = SIMD_DIV_P(uu,dd);
    ww      = SIMD_DIV_P(w_reg[i],dd);
    temp_x  = SIMD_SUB_P(x_reg[i],SIMD_MUL_P(ss, xx1));
    xx      = SIMD_SUB_P(temp_x,SIMD_MUL_P(ll, xx));
    xx      = SIMD_DIV_P(xx,dd);
    u2[i] = uu;
    w2[i] = ww;
    x2[i] = xx;
  }
  xx1     = x2[SIMD_VEC-2];
  uu1     = u2[SIMD_VEC-2];
  ww1     = w2[SIMD_VEC-2];

  for (int n = SIMD_VEC; n < ROUND_DOWN(sys_size, SIMD_VEC); n += SIMD_VEC) {
    LOAD(s_reg, s, n, sys_pad);
    LOAD(l_reg, l, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    LOAD(w_reg, w, n, sys_pad);
    LOAD(x_reg, x, n, sys_pad);
    
    for (int i = 0; i < SIMD_VEC; i++) {  
    ss      = s_reg[i];
    ll      = SIMD_SUB_P(l_reg[i], SIMD_MUL_P(ss, uu1));
    temp_d  = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(ss, ww1));
    dd      = SIMD_SUB_P(temp_d, SIMD_MUL_P(ll, uu));
    uu1     = uu;
    uu      = SIMD_SUB_P(u_reg[i], SIMD_MUL_P(ll, ww));
    uu      = SIMD_DIV_P(uu,dd);
    ww1     = ww;
    ww      = SIMD_DIV_P(w_reg[i],dd);
    temp_x  = SIMD_SUB_P(x_reg[i],SIMD_MUL_P(ss, xx1));
    xx1     = xx;
    xx      = SIMD_SUB_P(temp_x,SIMD_MUL_P(ll, xx));
    xx      = SIMD_DIV_P(xx,dd);
    u2[n + i] = uu;
    w2[n + i] = ww;
    x2[n + i] = xx;
    }
  }

  // forward on remainder

  if (sys_size != sys_pad) {
    // perform a noncomplete forward
    // Loads are safe since sys_pads must be a multiple of SIMD_WIDTH, and
    // data in the padding is never used

    int n = ROUND_DOWN(sys_size, SIMD_VEC);
    LOAD(s_reg, s, n, sys_pad);
    LOAD(l_reg, l, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    LOAD(w_reg, w, n, sys_pad);
    LOAD(x_reg, x, n, sys_pad);
    for (int i = 0; (n + i) < sys_size; i++) {

    ss      = s_reg[i];
    ll      = SIMD_SUB_P(l_reg[i], SIMD_MUL_P(ss, uu1));
    temp_d  = SIMD_SUB_P(d_reg[i], SIMD_MUL_P(ss, ww1));
    dd      = SIMD_SUB_P(temp_d, SIMD_MUL_P(ll, uu));
    uu1     = uu;
    uu      = SIMD_SUB_P(u_reg[i], SIMD_MUL_P(ll, ww));
    uu      = SIMD_DIV_P(uu,dd);
    ww1     = ww;
    ww      = SIMD_DIV_P(w_reg[i],dd);
    temp_x  = SIMD_SUB_P(x_reg[i],SIMD_MUL_P(ss, xx1));
    xx1     = xx;
    xx      = SIMD_SUB_P(temp_x,SIMD_MUL_P(ll, xx));
    xx      = SIMD_DIV_P(xx,dd);
    u2[n + i] = uu;
    w2[n + i] = ww;
    x2[n + i] = xx;
    }
  }

  // backward on last chunk
  int n = ROUND_DOWN(sys_size, SIMD_VEC);
  if (sys_size != sys_pad) {
    x_reg[sys_size - 1 - n] = xx;
    xx1 = SIMD_SUB_P(xx1, SIMD_MUL_P(uu1, xx));
    x_reg[sys_size - 2 - n] = xx1;
    for (int i = sys_size - n - 3; i >= 0; i--) {
      temp_x  = SIMD_SUB_P(x2[n + i], SIMD_MUL_P(w2[n + i], xx));
      xx = xx1;
      xx1 =  SIMD_SUB_P(temp_x, SIMD_MUL_P(u2[n + i], xx1));
      x_reg[i] = xx1;
    }

    STORE(x, x_reg, n, sys_pad);
  } else {
    x_reg[SIMD_VEC - 1] = xx;
    xx1 = SIMD_SUB_P(xx1, SIMD_MUL_P(uu1, xx));
    x_reg[SIMD_VEC - 2] = xx1;
    n -= SIMD_VEC;
    for (int i = SIMD_VEC - 3; i >= 0; i--) {
      temp_x  = SIMD_SUB_P(x2[n + i], SIMD_MUL_P(w2[n + i], xx));
      xx = xx1;
      xx1 =  SIMD_SUB_P(temp_x, SIMD_MUL_P(u2[n + i], xx1));
      x_reg[i] = xx1;

    }

      STORE(x, x_reg, n, sys_pad);
  }
  n -= SIMD_VEC;

  //
  // backward pass
  //

  for (; n >= 0; n -= SIMD_VEC) {
    for (int i = (SIMD_VEC - 1); i >= 0; i--) {
      temp_x  = SIMD_SUB_P(x2[n + i], SIMD_MUL_P(w2[n + i], xx));
      xx = xx1;
      xx1 =  SIMD_SUB_P(temp_x, SIMD_MUL_P(u2[n + i], xx1));
      x_reg[i] = xx1;     
    }
      STORE(x, x_reg, n, sys_pad);
  }
}
template <typename REAL>
void pentad_scalar(const REAL *__restrict s, const REAL *__restrict l,
                 const REAL *__restrict d, const REAL *__restrict u, const REAL *__restrict w,
                 REAL *__restrict x, REAL *__restrict y, int N, int stride, bool inc) {
  int i = 0;
  int ind = 0;
  REAL ss, ll, dd, uu, ww, xx,xx1;
  std::array<REAL, N_MAX> u2{};
  std::array<REAL, N_MAX> w2{}; 
  std::array<REAL, N_MAX> x2{};
  //
  // forward pass
  //
  //row 1
  dd    = 1.0F / d[0];
  uu    = dd * u[0];
  ww    = dd * w[0];
  xx    = dd * x[0];
  u2[0] = uu;
  w2[0] = ww;
  x2[0] = xx;

  //row 2
  ind   = ind + stride;
  ll = l[ind];
  dd = d[ind] - ll *uu;
  u2[1]    = (u[ind] - ll * ww) / dd;
  w2[1]    =  w[ind] / dd;
  x2[1]     = (x[ind] - ll * xx) / dd;

  for (i = 2; i < N; i++) {
    ind   = ind + stride;
    ss    = s[ind];
    ll    = l[ind] - ss * u2[i - 2];
    dd    = d[ind] - ss * w2[i - 2] - ll * u2[i - 1];
    u2[i]    = (u[ind] - ll * w2[i - 1]) / dd;
    w2[i]    = w[ind] / dd;
    xx    = (x[ind] - ss * x2[i - 2] - ll * x2[i - 1]) / dd;
    x2[i]  = xx;
  }
  //
  // reverse pass
  x[ind] = xx;
  ind = ind - stride;
  uu  = u2[N - 2];
  xx1 = x2[N - 2] - uu * xx;
  x[ind] = xx1;
  
  for (i = N - 3; i >= 0; i--) {
    ind = ind - stride;
    ww = w2[i];
    uu = u2[i];
    REAL temp =  x2[i] - ww * xx - uu * xx1;
    xx = xx1;
    xx1 = temp;
    x[ind] = xx1;
  }
}
//
// pentadiagonal solver; vectorised solution where the system dimension is not
// the same as the vectorisation dimension
//
template <typename REAL, typename VECTOR>
void pentad_scalar_vec(const REAL *__restrict h_s, const REAL *__restrict h_l, const REAL *__restrict h_d, const REAL *__restrict h_u, const REAL *__restrict h_w, REAL *__restrict h_x, REAL *__restrict h_y, int N, int stride, bool inc) {

  int i, ind = 0;
  VECTOR ss, ll, dd, uu, uu1, ww, ww1, xx, xx1,u2[N_MAX], w2[N_MAX], x2[N_MAX];

  //
  // forward pass
  //
  //row 0
  dd    = SIMD_LOAD_P(&h_d[0]);
  uu    = SIMD_LOAD_P(&h_u[0]) / dd;
  ww    = SIMD_LOAD_P(&h_w[0]) / dd;
  xx    = SIMD_LOAD_P(&h_x[0]) / dd;
  u2[0] = uu;
  w2[0] = ww;
  x2[0] = xx;
  //row 1
  ind   = ind + stride;
  ll    = SIMD_LOAD_P(&h_l[ind]);
  dd    = SIMD_LOAD_P(&h_d[ind]) - ll * uu;
  uu    = (SIMD_LOAD_P(&h_u[ind]) - ll * ww) / dd;
  ww    = SIMD_LOAD_P(&h_w[ind]) / dd;
  xx    = (SIMD_LOAD_P(&h_x[ind]) - ll * xx) / dd;
  u2[1] = uu;
  w2[1] = ww;
  x2[1] = xx;


  for (i = 2; i < N; i++) {
    ind   = ind + stride;
    xx1   = x2[i-2];
    uu1   = u2[i-2];
    ww1   = w2[i-2];
    ss    = SIMD_LOAD_P(&h_s[ind]);
    ll    = SIMD_LOAD_P(&h_l[ind]) - ss * uu1;
    dd    = SIMD_LOAD_P(&h_d[ind]) - ss * ww1 - ll * uu;
    uu    = (SIMD_LOAD_P(&h_u[ind]) - ll * ww) / dd;
    ww    = SIMD_LOAD_P(&h_w[ind]) / dd;
    xx    = (SIMD_LOAD_P(&h_x[ind]) - ss * xx1 - ll * xx)/dd;
    u2[i] = uu;
    w2[i] = ww;
    x2[i] = xx;
  }
  //
  // reverse pass
  //
  if (inc)
  {
    //row 0
    SIMD_STORE_P(&h_x[ind], xx);
    //row 1
    ind = ind - stride;
    uu  = u2[N-2];
    xx1  = x2[N-2] - uu *xx;
    SIMD_STORE_P(&h_x[ind], xx1);
  }
    
  else{
    //row 0
     xx = x2[N-1];
    SIMD_STORE_P(&h_x[ind], xx);
    //row 1
    ind = ind - stride;
    uu  = u2[N-2];
    xx1  = x2[N-2] - uu *xx;
    SIMD_STORE_P(&h_x[ind], xx1);
  } 
  for (i = N - 3; i >= 0; i--) {
    ind = ind - stride;
    ww  = w2[i];
    uu  = u2[i];
    VECTOR temp =  x2[i] - ww * xx - uu * xx1;
    xx = xx1;
    xx1 = temp;
    if (inc)
      SIMD_STORE_P(&h_x[ind], xx1);
      ///SIMD_STORE_P(&h_u[ind], SIMD_LOAD_P(&h_u[ind]) + dd);
    else
      SIMD_STORE_P(&h_x[ind], xx1);
  }
}

template <typename REAL>
void pentadsolver_x(const REAL *__restrict ds, const REAL *__restrict dl, const REAL *__restrict d,
                    const REAL *__restrict du, const REAL *__restrict dw, REAL *__restrict x,
                    size_t t_sys_size) {
  constexpr size_t N_MAX = 1024; // FIXME move to parameter, define
  std::array<REAL, N_MAX> du2{};
  std::array<REAL, N_MAX> dw2{};
  std::array<REAL, N_MAX> x2{};
  // row 1 - normalise -- ds, dl 0
  du2[0] = du[0] / d[0];
  dw2[0] = dw[0] / d[0];
  x2[0]  = x[0] / d[0];

  // row 2 - /-1 , normalise - ds 0
  REAL ddl = dl[1];
  REAL dd  = d[1] - ddl * du2[0];
  du2[1]    = (du[1] - ddl * dw2[0]) / dd;
  dw2[1]    = dw[1] / dd;
  x2[1]     = (x[1] - ddl * x2[0]) / dd;

  // rest
  #pragma omp simd
  for (size_t i = 2; i < t_sys_size; ++i) {
    // row i - (dds*row{i-2}) - ddu'-row{i-1}, normalise
    // TODO: check with Istvan -- ds, dl, du, dw requirements -- last elements
    // must be 0, but are they accessible? remove comp du2[N-1], dw2[N-2..N-1]
    REAL dds = ds[i];
    ddl       = dl[i] - dds * du2[i - 2];
    dd        = d[i] - dds * dw2[i - 2] - ddl * du2[i - 1];
    du2[i]    = (du[i] - ddl * dw2[i - 1]) / dd;
    dw2[i]    = dw[i] / dd;
    x2[i]     = (x[i] - dds * x2[i - 2] - ddl * x2[i - 1]) / dd;
  }
  //
  // Backward substitution
  //
  // row t_sys_size - 1
  x[t_sys_size - 1] = x2[t_sys_size - 1];
  // row t_sys_size - 2
  REAL ddu         = du2[t_sys_size - 2];
  x[t_sys_size - 2] = x2[t_sys_size - 2] - ddu * x[t_sys_size - 1];
  // rest
  //#pragma omp simd
  for (int i = static_cast<int>(t_sys_size) - 3; i >= 0; --i) {
    // row i - (ddw*row{i+2}) - ddu-row{i+1}
    REAL ddw = dw2[i];
    ddu       = du2[i];
    x[i]      = x2[i] - ddw * x[i + 2] - ddu * x[i + 1];
  }
}


template <typename REAL>
void pentadsolver_gpsv_batch_x(const REAL *ds, const REAL *dl, const REAL *d,
                               const REAL *du, const REAL *dw, REAL *x, REAL *y,
                               const int *t_dims, size_t t_ndims,
                               void * /*t_buffer*/, const int *pads, bool inc) {
  size_t n_sys =
      std::accumulate(t_dims + 1, t_dims + t_ndims, 1, std::multiplies<>());
  size_t sys_size = t_dims[0];

  int sys_stride = 1;
  int sys_pads = pads[0];
  
  if(t_ndims == 3){
    if(sys_pads % SIMD_VEC == 0){
  #pragma omp parallel for collapse(2)
      for (int k = 0; k < t_dims[2]; k++) {
        for (int j = 0; j < ROUND_DOWN(t_dims[1], SIMD_VEC); j += SIMD_VEC) {
          int ind = k * pads[0] * pads[1] + j * pads[0];
          pentad_x_transpose(&ds[ind], &dl[ind], &d[ind], &du[ind],
                                      &dw[ind], &x[ind], sys_size, sys_pads);
        }
      }
      if (ROUND_DOWN(t_dims[1], SIMD_VEC) <
          t_dims[1]) { // If there is leftover, fork threads an compute it
      #pragma omp parallel for collapse(2)
        for (int k = 0; k < t_dims[2]; k++) {
          for (int j = ROUND_DOWN(t_dims[1], SIMD_VEC); j < t_dims[1]; j++) {
            int ind = k * pads[0] * pads[1] + j * pads[0];
            pentad_scalar(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
          }
        }
      }
    }
    else{
      #pragma omp parallel for collapse(2)
      for (int k = 0; k < t_dims[2]; k++) {
        for (int j = 0; j < t_dims[1]; j++) {
          int ind = k * pads[0] * pads[1] + j * pads[0];
          pentad_scalar(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
        }
      }
    }

  }
  
  else if (t_ndims == 2)
  {
    if(sys_pads % SIMD_VEC == 0){
    #pragma omp parallel for
        for (int j = 0; j < ROUND_DOWN(t_dims[1], SIMD_VEC); j += SIMD_VEC) {
          int ind = j * pads[0];
          pentad_x_transpose(&ds[ind], &dl[ind], &d[ind], &du[ind],
                                      &dw[ind], &x[ind], sys_size, sys_pads);
        }      
      if (ROUND_DOWN(t_dims[1], SIMD_VEC) <
          t_dims[1]) { // If there is leftover, fork threads an compute it
        #pragma omp parallel for       
          for (int j = ROUND_DOWN(t_dims[1], SIMD_VEC); j < t_dims[1]; j++) {
            int ind = j * pads[0];
            pentad_scalar(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
          }       
      }
    }
    else{
        #pragma omp parallel for
        for (int j = 0; j < t_dims[1]; j++) {
          int ind = j * pads[0];
          pentad_scalar(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
        }
      }

  }
  else{
#pragma omp parallel for
  for (int i = 0; i < n_sys; ++i) {
    size_t sys_start = i * sys_size;
    pentadsolver_x(ds + sys_start, dl + sys_start, d + sys_start,
                   du + sys_start, dw + sys_start, x + sys_start, sys_size);
  }
  }
 }                   
template <typename REAL>
void pentadsolver_gpsv_batch_outermost(const REAL *ds, const REAL *dl,
                                    const REAL *d, const REAL *du,
                                    const REAL *dw, REAL *x, REAL *y,
                                    const int *t_dims, size_t t_ndims,
                                    int t_solvedim, void * /*t_buffer*/, const int *pads, bool inc) {
  // size_t n_sys =
  size_t sys_size = t_dims[t_ndims - 1];
  int sys_stride = pads[0] * pads[1];         //Stride between the consecutive elements of the system
  #pragma omp parallel for collapse(2)       // Interleaved scheduling for better data
                                            // locality and thus lower TLB miss rate
    for (int j = 0; j < t_dims[1]; j++) {
      for (int i = 0; i < ROUND_DOWN(t_dims[0], SIMD_VEC); i += SIMD_VEC) {
        int ind = j * pads[0] + i;
        pentad_scalar_vec<REAL, VECTOR>(&ds[ind], &dl[ind], &d[ind], 
        &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
      }
    }
    if (ROUND_DOWN(t_dims[0], SIMD_VEC) <
        t_dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int j = 0; j < t_dims[1]; j++) {
        for (int i = ROUND_DOWN(t_dims[0], SIMD_VEC); i < t_dims[0]; i++) {
          int ind = j * pads[0] + i;
          pentad_scalar<REAL>(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
        }
      }
    }
   
  
}

template <typename REAL>
void pentadsolver_gpsv_batch_middle(const REAL *ds, const REAL *dl,
                                    const REAL *d, const REAL *du,
                                    const REAL *dw, REAL *x, REAL *y,
                                    const int *t_dims, size_t t_ndims,
                                    int t_solvedim, void * /*t_buffer*/, const int *pads, bool inc) {
  size_t n_sys_in =
      std::accumulate(t_dims, t_dims + t_solvedim, 1, std::multiplies<>());
  size_t n_sys_out = std::accumulate(t_dims + t_solvedim + 1, t_dims + t_ndims, 1, std::multiplies<>());
  size_t sys_size  = t_dims[t_solvedim];
 
  int sys_stride = pads[0]; // Stride between the consecutive elements of a system
  if (t_ndims == 3){
  #pragma omp parallel for collapse(2)
    for (int k = 0; k < t_dims[2]; k++) {
      for (int i = 0; i < ROUND_DOWN(t_dims[0], SIMD_VEC); i += SIMD_VEC) {
        int ind = k * pads[0] * pads[1] + i;
        pentad_scalar_vec<REAL, VECTOR>(&ds[ind], &dl[ind], &d[ind], 
        &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
      }
    }
    if (ROUND_DOWN(t_dims[0], SIMD_VEC) <
        t_dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for collapse(2)
      for (int k = 0; k < t_dims[2]; k++) {
        for (int i = ROUND_DOWN(t_dims[0], SIMD_VEC); i < t_dims[0]; i++) {
          int ind = k * pads[0] * pads[1] + i;
          pentad_scalar<REAL>(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
        }
      }
    }
  }
  else{
    #pragma omp parallel for
      for (int i = 0; i < ROUND_DOWN(t_dims[0], SIMD_VEC); i += SIMD_VEC) {
        int ind = i;
        pentad_scalar_vec<REAL, VECTOR>(&ds[ind], &dl[ind], &d[ind], 
        &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
      }
    if (ROUND_DOWN(t_dims[0], SIMD_VEC) <
        t_dims[0]) { // If there is leftover, fork threads an compute it
#pragma omp parallel for
        for (int i = ROUND_DOWN(t_dims[0], SIMD_VEC); i < t_dims[0]; i++) {
          int ind = i;
          pentad_scalar<REAL>(&ds[ind], &dl[ind], &d[ind], &du[ind], &dw[ind], &x[ind], &y[ind], sys_size, sys_stride, inc);
        }
    }

  
  }
  
}

template <typename REAL>
void pentadsolver_gpsv_batch(const REAL *ds, const REAL *dl, const REAL *d,
                             const REAL *du, const REAL *dw, REAL *x, const int *t_dims, size_t t_ndims, int t_solvedim,
                             void *t_buffer) {

  int pads[3] = {1,1,1};
  pads[0] = (1+((t_dims[0]-1)/SIMD_VEC))*SIMD_VEC; // Compute padding for vecotrization
  pads[1] = t_dims[1], pads[2] = t_dims[2];
  REAL *y = nullptr;
  if (t_solvedim == 0) {
    BEGIN_PROFILING("pentad_x");
    pentadsolver_gpsv_batch_x(ds, dl, d, du, dw, x, y, t_dims, t_ndims, t_buffer, pads, false);
    END_PROFILING("pentad_x");
  } else if (t_solvedim == 2) {
    pentadsolver_gpsv_batch_outermost(ds, dl, d, du, dw, x, y, t_dims, t_ndims, t_solvedim, t_buffer, pads, true);
  } else {
    BEGIN_PROFILING("pentad_y");
    pentadsolver_gpsv_batch_middle(ds, dl, d, du, dw, x, y, t_dims, t_ndims, t_solvedim, t_buffer, pads, false);
    END_PROFILING("pentad_y");
    PROFILE_REPORT();
  }
}

// ----------------------------------------------------------------------------
// Pentadsolver context functions
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Buffer size calculation
// ----------------------------------------------------------------------------

template <typename Float>
[[nodiscard]] size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t /*handle*/, const Float * /*ds*/,
    const Float * /*dl*/, const Float * /*d*/, const Float * /*du*/,
    const Float * /*dw*/, const Float * /*x*/, const int * /*t_dims*/,
    size_t /*t_ndims*/, int /*t_solvedim*/) {
  return 0;
}

// ----------------------------------------------------------------------------
// Adapter function implementations
// ----------------------------------------------------------------------------
#if FPPREC == 1
size_t pentadsolver_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, const double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_D_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const double *ds, const double *dl,
    const double *d, const double *du, const double *dw, double *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}
#elif FPPREC == 0
size_t pentadsolver_gpsv_batch_buffer_size_ext(pentadsolver_handle_t handle,
                                               const float *ds, const float *dl,
                                               const float *d, const float *du,
                                               const float *dw, const float *x,
                                               const int *t_dims, int t_ndim,
                                               int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(
      handle, ds, dl, d, du, dw, x, t_dims, static_cast<size_t>(t_ndim),
      t_solvedim);
}

size_t pentadsolver_S_gpsv_batch_buffer_size_ext(
    pentadsolver_handle_t handle, const float *ds, const float *dl,
    const float *d, const float *du, const float *dw, float *x,
    const int *t_dims, int t_ndim, int t_solvedim) {
  return pentadsolver_gpsv_batch_buffer_size_ext(handle, ds, dl, d, du, dw, x,
                                                 t_dims, t_ndim, t_solvedim);
}
#endif
#if FPPREC == 1
void pentadsolver_gpsv_batch(pentadsolver_handle_t /*handle*/, const double *ds,
                             const double *dl, const double *d,
                             const double *du, const double *dw, double *x,
                             const int *t_dims, int t_ndim, int t_solvedim,
                             void *t_buffer) {
  pentadsolver_gpsv_batch(ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_D_gpsv_batch(pentadsolver_handle_t handle, const double *ds,
                               const double *dl, const double *d,
                               const double *du, const double *dw, double *x,
                               const int *t_dims, int t_ndim, int t_solvedim,
                               void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}
#elif FPPREC == 0
void pentadsolver_gpsv_batch(pentadsolver_handle_t /*handle*/, const float *ds,
                             const float *dl, const float *d, const float *du,
                             const float *dw, float *x, const int *t_dims,
                             int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(ds, dl, d, du, dw, x, t_dims,
                          static_cast<size_t>(t_ndim), t_solvedim, t_buffer);
}

void pentadsolver_S_gpsv_batch(pentadsolver_handle_t handle, const float *ds,
                               const float *dl, const float *d, const float *du,
                               const float *dw, float *x, const int *t_dims,
                               int t_ndim, int t_solvedim, void *t_buffer) {
  pentadsolver_gpsv_batch(handle, ds, dl, d, du, dw, x, t_dims, t_ndim,
                          t_solvedim, t_buffer);
}
#  endif