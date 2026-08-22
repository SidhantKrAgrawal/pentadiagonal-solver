#ifndef PENTADSOLVER_DEVICE_MESH_HPP_INCLUDED
#define PENTADSOLVER_DEVICE_MESH_HPP_INCLUDED

#include "mesh.hpp"

template <typename Float> class DeviceMesh : public Mesh<Float> {
  Float *_ds_d = nullptr;
  Float *_dl_d = nullptr;
  Float *_d_d  = nullptr;
  Float *_du_d = nullptr;
  Float *_dw_d = nullptr;
  Float *_x_d  = nullptr;

public:
  explicit DeviceMesh(const std::filesystem::path &file_name);
  DeviceMesh(size_t solve_dim, std::vector<int> _dims);
  ~DeviceMesh() {
    cudaFree(_ds_d);
    cudaFree(_dl_d);
    cudaFree(_d_d);
    cudaFree(_du_d);
    cudaFree(_dw_d);
    cudaFree(_x_d);
  }

  const Float *ds_d() const { return _ds_d; }
  const Float *dl_d() const { return _dl_d; }
  const Float *d_d() const { return _d_d; }
  const Float *du_d() const { return _du_d; }
  const Float *dw_d() const { return _dw_d; }
  const Float *x_d() const { return _x_d; }
  Float *x_d() { return _x_d; }
};

template <typename Float>
DeviceMesh<Float>::DeviceMesh(const std::filesystem::path &file_name)
    : Mesh<Float>::Mesh(file_name) {
  auto init_arr = [](Float **arr_d, const std::vector<Float> &arr_h) {
    cudaMalloc((void **)arr_d, sizeof(Float) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(Float) * arr_h.size(),
               cudaMemcpyHostToDevice);
  };
  init_arr(&_ds_d, this->_ds);
  init_arr(&_dl_d, this->_dl);
  init_arr(&_d_d, this->_d);
  init_arr(&_du_d, this->_du);
  init_arr(&_dw_d, this->_dw);
  init_arr(&_x_d, this->_x);
}

template <typename Float>
DeviceMesh<Float>::DeviceMesh(size_t solve_dim, std::vector<int> dims)
    : Mesh<Float>::Mesh(solve_dim, dims) {
  auto init_arr = [](Float **arr_d, const std::vector<Float> &arr_h) {
    cudaMalloc((void **)arr_d, sizeof(Float) * arr_h.size());
    cudaMemcpy(*arr_d, arr_h.data(), sizeof(Float) * arr_h.size(),
               cudaMemcpyHostToDevice);
  };
  init_arr(&_ds_d, this->_ds);
  init_arr(&_dl_d, this->_dl);
  init_arr(&_d_d, this->_d);
  init_arr(&_du_d, this->_du);
  init_arr(&_dw_d, this->_dw);
  init_arr(&_x_d, this->_x);
}

#endif /* ifndef PENTADSOLVER_DEVICE_MESH_HPP_INCLUDED */
