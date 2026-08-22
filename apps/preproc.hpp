
#include "pentad_common.hpp"
#include "pentad_simd.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

// solve u_momentum and v_momentum equations
template <typename REAL>
void uv_momentum(int nx, int ny, REAL dx, REAL dy, REAL Re,
  std::vector<std::vector<REAL>> &p, REAL alphaU,
  std::vector<std::vector<REAL>> &u_star,
  std::vector<std::vector<REAL>> &v_star,
  std::vector<std::vector<REAL>> &u_prime,
  std::vector<std::vector<REAL>> &v_prime,
  std::vector<std::vector<REAL>> &A_e,
  std::vector<std::vector<REAL>> &A_w,
  std::vector<std::vector<REAL>> &A_n,
  std::vector<std::vector<REAL>> &A_s,
  std::vector<std::vector<REAL>> &A_p,
  std::vector<std::vector<REAL>> &X_vec,
  std::vector<std::vector<REAL>> &Y_vec){

double D_e, D_w, D_s, D_n, F_e, F_w, F_s, F_n, pressure, temp;


D_e = dy / (dx*Re);
D_w = dy / (dx*Re);
D_n = dx / (dy*Re);
D_s = dx / (dy*Re);

for (int i = 2; i < ny; i++) {
for (int j = 1; j <= nx; j++) {  

F_e =dy*u_prime[i][j];
F_w =dy*u_prime[i][j - 1];  
F_n =dx*v_prime[i- 1][j];
F_s =dx*v_prime[i][j]; 

if(j == 1)
{
    A_e[i][j]=D_e +   std::max(0.0,-F_e)  ;
    A_w[i][j]=2*D_w + std::max(0.0,F_w)   ;
    A_n[i][j]=D_n +   std::max(0.0,-F_n)  ;
    A_s[i][j]=D_s +   std::max(0.0,F_s) ;
    A_p[i][j]=A_e[i][j] + A_w[i][j] + A_n[i][j] + A_s[i][j] + (F_e - F_w) + (F_n - F_s);

    X_vec[i][j]=0.5*(p[i][j]- p[i][j+1])*dx ;
    Y_vec[i][j]=0.5*(p[i+1][j] - p[i-1][j])*dy;

}

else if(j == nx)
{ 
  A_e[i][j]=D_e +   std::max(0.0,-F_e);
  A_w[i][j]=2*D_w + std::max(0.0,F_w);
  A_n[i][j]=D_n +   std::max(0.0,-F_n);
  A_s[i][j]=D_s +   std::max(0.0,F_s);
  A_p[i][j]=A_e[i][j] + A_w[i][j] + A_n[i][j] + A_s[i][j] + (F_e - F_w) + (F_n - F_s);

  X_vec[i][j]=0.5*(p[i][j-1] - p[i][j])*dx  ;
  Y_vec[i][j]=0.5*(p[i+1][j] - p[i-1][j])*dy;



}
else{

A_e[i][j] =D_e + std::max(0.0,-F_e);
A_w[i][j] =D_w + std::max(0.0,F_w);
A_n[i][j] =D_n + std::max(0.0,-F_n);
A_s[i][j] =D_s + std::max(0.0,F_s);
A_p[i][j] =A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

X_vec[i][j]=0.5*(p[i][j-1]-p[i][j+1])*dx;
Y_vec[i][j]=0.5*(p[i+1][j]-p[i-1][j])*dy;
}

}
}
//top wall and bottom wall
int i = 1;
int j;
for(j=2;j<nx;j++)
{
F_e =dy*u_prime[i][j];
F_w =dy*u_prime[i][j - 1];  
F_n =dx*v_prime[i- 1][j];
F_s =dx*v_prime[i][j]; 

A_e[i][j]=D_e +   std::max(0.0,-F_e);
A_w[i][j]=D_w +   std::max(0.0,F_w);
A_n[i][j]=2*D_n + std::max(0.0,-F_n);
A_s[i][j]=D_s +   std::max(0.0,F_s);
A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

        X_vec[i][j]=0.5*(p[i][j-1] - p[i][j+1])*dx;
        Y_vec[i][j]=0.5*(p[i+1][j] - p[i][j])*dy ;

}
i=ny;
for(j=2;j<nx;j++)
{
    
    F_e =dy*u_prime[i][j];
    F_w =dy*u_prime[i][j - 1];  
    F_n =dx*v_prime[i- 1][j];
    F_s =dx*v_prime[i][j]; 

    A_e[i][j]=D_e +   std::max(0.0,-F_e);
    A_w[i][j]=D_w +   std::max(0.0,F_w);
    A_n[i][j]=D_n +   std::max(0.0,-F_n);
    A_s[i][j]=2*D_s + std::max(0.0,F_s);
    A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

    X_vec[i][j]=0.5*(p[i][j-1] - p[i][j+1])*dx;
    Y_vec[i][j]=0.5*(p[i][j] - p[i-1][j])*dy ;
  
   
}
//Corner BC's

//top left
  i=1;
  j=1;


  F_e =dy*u_prime[i][j];
  F_w =dy*u_prime[i][j - 1];  
  F_n =dx*v_prime[i- 1][j];
  F_s =dx*v_prime[i][j];

  A_e[i][j]=D_e +   std::max(0.0,-F_e);
  A_w[i][j]=2*D_w + std::max(0.0,F_w);
  A_n[i][j]=2*D_n + std::max(0.0,-F_n);
  A_s[i][j]=D_s +   std::max(0.0,F_s);
  A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

    X_vec[i][j]=0.5*(p[i][j] - p[i][j+1])*dx ;
    Y_vec[i][j]=0.5*(p[i+1][j] - p[i][j])*dy  ;

  //top right
  i=1;
  j=nx;
  
  
  F_e =dy*u_prime[i][j];
  F_w =dy*u_prime[i][j - 1];  
  F_n =dx*v_prime[i- 1][j];
  F_s =dx*v_prime[i][j];

  A_e[i][j]=D_e +   std::max(0.0,-F_e);
  A_w[i][j]=2*D_w + std::max(0.0,F_w);
  A_n[i][j]=2*D_n + std::max(0.0,-F_n);
  A_s[i][j]=D_s +   std::max(0.0,F_s);
  A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

  X_vec[i][j]=0.5*(p[i][j-1] - p[i][j])*dx ; 
  Y_vec[i][j]=0.5*(p[i+1][j] - p[i][j])*dy ; 

  //bottom left
  i=ny;
  j=1;


  F_e =dy*u_prime[i][j];
  F_w =dy*u_prime[i][j - 1];  
  F_n =dx*v_prime[i- 1][j];
  F_s =dx*v_prime[i][j];

  A_e[i][j]=D_e +   std::max(0.0,-F_e);
  A_w[i][j]=2*D_w + std::max(0.0,F_w);
  A_n[i][j]=D_n +   std::max(0.0,-F_n);
  A_s[i][j]=2*D_s + std::max(0.0,F_s);
  A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

    X_vec[i][j]=0.5*(p[i][j] - p[i][j+1])*dx ; 
    Y_vec[i][j]=0.5*(p[i][j] - p[i-1][j])*dy  ;


//bottom right
  i=ny;
  j=nx;


  F_e =dy*u_prime[i][j];
  F_w =dy*u_prime[i][j - 1];  
  F_n =dx*v_prime[i- 1][j];
  F_s =dx*v_prime[i][j];

  A_e[i][j]=2*D_e + std::max(0.0,-F_e);
  A_w[i][j]=2*D_w +   std::max(0.0,F_w);
  A_n[i][j]=D_n +     std::max(0.0,-F_n);
  A_s[i][j]=D_s +     std::max(0.0,F_s);
  A_p[i][j]=A_w[i][j] + A_e[i][j] + A_n[i][j] + A_s[i][j] + (F_e-F_w) + (F_n-F_s);

  X_vec[i][j]=0.5*(p[i][j-1] - p[i][j])*dx;
  Y_vec[i][j]=0.5*(p[i][j] - p[i-1][j])*dy;
}

template <typename REAL>
void uv_prime(int nx, int ny, REAL dx, REAL dy, REAL alphaU,
  std::vector<std::vector<REAL>> &u,
  std::vector<std::vector<REAL>> &v,
  std::vector<std::vector<REAL>> &u_prime,
  std::vector<std::vector<REAL>> &v_prime,
  std::vector<std::vector<REAL>> &p, 
  std::vector<std::vector<REAL>> &uv, int iter) {
  
  int i, j;
  double temp_u1, temp_u2, temp_v1, temp_v2;

  for(i=1;i<ny+1;i++)
  {
    for (j=1;j<nx;j++)
    {
      temp_u1 = 0.5*(u[i][j] + u[i][j + 1])+ 
                0.25*alphaU*(p[i][j + 1] - p[i][j - 1])*dy/uv[i][j] + 
                0.25*alphaU*(p[i][j + 2] - p[i][j])*dy/uv[i][j + 1];
      
      temp_u2 = 0.5*alphaU*(1/uv[i][j] + 1/uv[i][j])*(p[i][j + 1] - 
                p[i][j])*dy;
     
      u_prime[i][j] = temp_u1 - temp_u2;
        
    }  
  }

  for(i=2;i<ny+1;i++)
  {
      for (j=1;j<nx+1;j++)
      {
          temp_v1 = 0.5*(v[i][j] + v[i - 1][j]) + 0.25*alphaU*(p[i - 1][j] - p[i + 1][j])*dy/uv[i][j] + 
                    0.25*alphaU*(p[i - 2][j] - p[i][j])*dy/uv[i-1][j];
          
          temp_v2  = 0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(p[i - 1][j] - p[i][j])*dy;
          
          v_prime[i-1][j]= temp_v1 - temp_v2;
      }
  }

}

//
// calculate r.h.s. and set penta-diagonal coefficients
//
template <typename REAL>
void pressure_coef_matrix(int nx, int ny, int nx_pad, REAL alphaU, REAL dx, REAL dy,
            std::vector<std::vector<REAL>> &u_prime,
            std::vector<std::vector<REAL>> &v_prime,
            std::vector<std::vector<REAL>> &uv,
            REAL *__restrict x, REAL *__restrict s_x,
            REAL *__restrict l_x, REAL *__restrict d_x,
            REAL *__restrict u_x, REAL *__restrict w_x,
            REAL *__restrict s_y, REAL *__restrict l_y,
            REAL *__restrict d_y, REAL *__restrict u_y,
            REAL *__restrict w_y) {

REAL d_s, d_l, d_d, d_u, d_w, xx;
int index,i,j;

  for(i=2; i < ny; i++)
  {
    for(j=1;j <= nx; j++)
    {
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

      if( j ==1){
      
      index =(i-1)*nx_pad ;

      d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
      d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
      d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
      d_d=d_u + d_l + d_s + d_w;

      xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;

    x[index]   = xx;
    s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w;
  }

     else if(j == nx){
      index=i*nx_pad -1;    

      d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
      d_s=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
      d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
      d_d = d_u + d_l + d_s + d_w;

      xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;

    x[index]   = xx;
    s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w;
  }
    else{

        index=(i-1)*nx_pad + (j-1);

        d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
        d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
        d_s=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
        d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
        d_d = d_u + d_l + d_s + d_w;

        xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;
          x[index]   = xx;
          s_x[index] = d_s;
          l_x[index] = d_l;
          d_x[index] = d_d;
          u_x[index] = d_u;
          w_x[index] = d_w;
          s_y[index] = d_s;
          l_y[index] = d_l;
          d_y[index] = d_d;
          u_y[index] = d_u;
          w_y[index] = d_w;
     }
    
  } 

  }
  
    //top
    i=1;
    for(j=2;j<nx;j++)
    { 
        index=(j-1);
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

        d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
        d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
        d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
        d_d=d_d=d_u + d_l + d_s + d_w;
        xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;

        x[index]   = xx;
        s_x[index] = d_s;
        l_x[index] = d_l;
        d_x[index] = d_d;
        u_x[index] = d_u;
        w_x[index] = d_w;
        s_y[index] = d_s;
        l_y[index] = d_l;
        d_y[index] = d_d;
        u_y[index] = d_u;
        w_y[index] = d_w;       
    }

    i=ny;
    for(j=2;j<nx;j++)
    { 
        index =(ny-1)*nx + (j-1);

        d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
        d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
        d_s=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
        d_d=d_d=d_u + d_l + d_s + d_w;

        xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;

        x[index]   = xx;
        s_x[index] = d_s;
        l_x[index] = d_l;
        d_x[index] = d_d;
        u_x[index] = d_u;
        w_x[index] = d_w;
        s_y[index] = d_s;
        l_y[index] = d_l;
        d_y[index] = d_d;
        u_y[index] = d_u;
        w_y[index] = d_w; 
    }
    
    //top left
    i=1;
    j=1;
    index=0;
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

    d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
    d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
    d_d=d_d=d_u + d_l + d_s + d_w;

    xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;
    
     x[index]   = xx;
     s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w; 
    
    
    
    //top right
    i=1;
    j=nx;
    index=nx-1;
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

    d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
    d_w=0.5*alphaU*(1/uv[i][j] + 1/uv[i+1][j])*(dx*dx);
    d_d=d_d=d_u + d_l + d_s + d_w;

    xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;
    x[index]   = xx;
    s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w;

    //bottom left
    i=ny;
    j=1;
    index=(ny-1)*nx ;
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

    d_u=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j+1])*(dy*dy);
    d_s=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
    d_d=d_d=d_u + d_l + d_s + d_w;
    
    xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;
    
    x[index]   = xx;
    s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w;


    //bottom right
    i=ny;
    j=nx;
    index=nx*ny-1;
        d_s     = 1;
        d_l     = 1;
        d_d     = 1;
        d_u     = 1;
        d_w     = 1;
        xx      = 1;

    d_l=0.5*alphaU*(1/uv[i][j] + 1/uv[i][j-1])*(dy*dy);
    d_s=0.5*alphaU*(1/uv[i][j] + 1/uv[i-1][j])*(dx*dx);
    d_d=d_d=d_u + d_l + d_s + d_w;

    xx =-(u_prime[i][j] - u_prime[i][j-1])*dy - (v_prime[i-1][j] - v_prime[i][j])*dx;
    
    x[index]   = xx;
    s_x[index] = d_s;
    l_x[index] = d_l;
    d_x[index] = d_d;
    u_x[index] = d_u;
    w_x[index] = d_w;
    s_y[index] = d_s;
    l_y[index] = d_l;
    d_y[index] = d_d;
    u_y[index] = d_u;
    w_y[index] = d_w;

}


template <typename REAL>
void pressure_correction(int nx, int ny, REAL alphaP, std::vector<std::vector<REAL>> &p, std::vector<std::vector<REAL>> &p_star, std::vector<std::vector<REAL>> &p_prime){
      
      for (int i = 1; i <= nx; ++i){
      p_prime[0][i] = p_prime[1][i];
      p_prime[ny + 1][i] = p_prime[ny][i];
      }

      for (int i = 1; i <= ny; ++i){
      p_prime[i][0] = p_prime[i][1];
      p_prime[i][nx + 1] = p_prime[i][nx];

      p_prime[0][0] = (p_prime[1][1] + p_prime[0][1] + p_prime[1][0]) / 3;
      p_prime[0][nx + 1] = (p_prime[0][nx] + p_prime[1][nx] + p_prime[1][nx + 1]) / 3;
      p_prime[ny + 1][0] = (p_prime[ny][0] + p_prime[ny][1] + p_prime[ny+ 1][1]) / 3;
      p_prime[ny + 1][nx + 1] = (p_prime[ny][nx + 1] + p_prime[ny + 1][nx] + p_prime[ny][nx]) / 3;
      
      for (int i = 0; i < ny + 2; ++i) {
      for (int j = 0; j < nx + 2; ++j) {
          p_star[i][j] = p[i][j] + (alphaP * p_prime[i][j]);

      }

  }
      
      }
}
template <typename REAL>
void velocity_correction(int nx, int ny, REAL alphaU, REAL dx, REAL dy,
          std::vector<std::vector<REAL>> &u,
          std::vector<std::vector<REAL>> &v,
          std::vector<std::vector<REAL>> &u_star,
          std::vector<std::vector<REAL>> &v_star,
          std::vector<std::vector<REAL>> &p_prime,
          std::vector<std::vector<REAL>> &uv) {
    int i,j;
    for(i=1;i<ny+1;i++) {
      for(j=1;j<=nx;j++) 
      {     
        if(j == 1){
        u_star[i][j]= u[i][j] + 0.5*alphaU*(p_prime[i][j] - p_prime[i][j+1])*dy/uv[i][j];
        }
        else if(j == nx){
          u_star[i][j]= u[i][j] + 0.5*alphaU*(p_prime[i][j-1] - p_prime[i][j])*dy/uv[i][j];
        }
        else{
        u_star[i][j]= u[i][j] + 0.5*alphaU*(p_prime[i][j-1] - p_prime[i][j+1])*dy/uv[i][j];
        }
      }       
  }

  for(i=1;i<=ny;i++) 
  {
      for(j=1;j<nx+1;j++) 
      {
          if(i == 1){
            v_star[i][j]=v[i][j] + 0.5*alphaU*(p_prime[i+1][j] - p_prime[i][j])*dx/uv[i][j];
          }
          if(i == ny){
            v_star[i][j]=v[i][j] + 0.5*alphaU*( p_prime[i][j] -  p_prime[i-1][j])*dx/uv[i][j];
          }
          else{
            v_star[i][j]=v[i][j] + 0.5*alphaU*(p_prime[i+1][j]-p_prime[i-1][j])*dx/uv[i][j];
          }
      }
  }

}


template <typename REAL>
void update_velocities(int nx, int ny, REAL alphaU, REAL dx, REAL dy,
          std::vector<std::vector<REAL>> &u_prime,
          std::vector<std::vector<REAL>> &v_prime,
          std::vector<std::vector<REAL>> &p_prime,
          std::vector<std::vector<REAL>> &uv){

  int i,j;
  for(i=1;i<ny+1;i++)
  { 
      for(j=1;j<nx;j++)
      {
          u_prime[i][j]=u_prime[i][j]+ 0.5*alphaU*(1/uv[i][j]+1/uv[i][j])*(p_prime[i][j]-p_prime[i][j+1])*dy;
      }
  }
  for(i=2;i<ny+1;i++)
  {
      for(j=1;j<nx+1;j++) 
      {
          v_prime[i-1][j]=v_prime[i-1][j] +  0.5*alphaU*(1/uv[i][j]+1/uv[i-1][j])*(p_prime[i][j]-p_prime[i-1][j])*dx;
      }
  }
}
template <typename REAL>
void update_values(int nx, int ny, REAL *__restrict sourceval, std::vector<std::vector<REAL>> &dest){
    int ind=0;

    for(int i=1;i<ny+1;i++)
    {
        for (int j=1;j<nx+1;j++)
        {
         
         dest[i][j]=sourceval[ind];
         ind++;
        }
    }
}

void solve(int nx, int ny,
           std::vector<std::vector<double>> &u,
           std::vector<std::vector<double>> &u_star,
           std::vector<std::vector<double>> &A_p,
           std::vector<std::vector<double>> &A_e,
          std::vector<std::vector<double>> &A_w,
          std::vector<std::vector<double>> &A_n,
          std::vector<std::vector<double>> &A_s,
          std::vector<std::vector<double>> &source_vec,
          double alphaU, double epsilonU, double &u_res)
 {

  int n;
    u_res = 0;
    for(int i =1; i<ny +1; i++)
    {
      for(int j =1; j<nx +1; j++)
      {
            u[i][j]= alphaU*(A_e[i][j]*u[i][j+1] + A_w[i][j]*u[i][j-1] + A_n[i][j]*u[i-1][j] + A_s[i][j]*u[i+1][j] + source_vec[i][j])/A_p[i][j] + (1-alphaU)*u_star[i][j];
          
      }
      
    }
    for(int i =1; i<ny+1; i++)
    {
      for(int j =1; j<nx+1; j++)
      {
        u_res+=std::pow((u[i][j] - alphaU*(A_e[i][j]*u[i][j+1] + A_w[i][j]*u[i][j-1] + A_n[i][j]*u[i-1][j] + A_s[i][j]*u[i+1][j] + source_vec[i][j])/A_p[i][j] - (1-alphaU)*u_star[i][j]),2);
        
      }

    }
  
}


