#ifndef PENTADSOLVER_MESH_HPP_INCLUDED
#define PENTADSOLVER_MESH_HPP_INCLUDED

#include <cmath>
#include <cassert>
#include <utility>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <omp.h>

template <typename Float> class Mesh {
protected:
  size_t _solve_dim;
  std::vector<int> _dims;
  std::vector<Float> _ds, _dl, _d, _du, _dw, _x, _u;

public:
  explicit Mesh(const std::filesystem::path &file_name);
  Mesh(size_t solve_dim, std::vector<int> _dims);

  [[nodiscard]] size_t solve_dim() const { return _solve_dim; }
  [[nodiscard]] const std::vector<int> &dims() const { return _dims; }
  const std::vector<Float> &ds() const { return _ds; }
  const std::vector<Float> &dl() const { return _dl; }
  const std::vector<Float> &d() const { return _d; }
  const std::vector<Float> &du() const { return _du; }
  const std::vector<Float> &dw() const { return _dw; }
  const std::vector<Float> &x() const { return _x; }
  const std::vector<Float> &u() const { return _u; }
  std::vector<Float> &x() { return _x; }
};
/**********************************************************************
 *                          Implementations                           *
 **********************************************************************/

template <typename Float>
inline void load_array(std::ifstream &f, size_t num_elements,
                       std::vector<Float> &array) {
  array.resize(num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
    // Load with the larger precision, then convert to the specified type
    f >> array[i];
  }
}

template <typename Float>
Mesh<Float>::Mesh(const std::filesystem::path &file_name)
    : _solve_dim{}, _ds{}, _dl{}, _d{}, _du{}, _dw{}, _x{}, _u{} {
  std::ifstream f(file_name);
  assert(f.good() && "Couldn't open file");
  size_t num_dims = 0;
  f >> num_dims >> _solve_dim;
  // Load sizes along the different dimensions
  size_t num_elements = 1;
  for (size_t i = 0; i < num_dims; ++i) {
    int size = 0;
    f >> size;
    _dims.push_back(size);
    num_elements *= size;
  }
  // Load arrays
  load_array(f, num_elements, _ds);
  load_array(f, num_elements, _dl);
  load_array(f, num_elements, _d);
  load_array(f, num_elements, _du);
  load_array(f, num_elements, _dw);
  load_array(f, num_elements, _x);
  if (std::is_same<Float, double>::value) {
    load_array(f, num_elements, _u);
  } else {
    std::string tmp;
    // Skip the line with the double values
    std::getline(f >> std::ws, tmp);
    load_array(f, num_elements, _u);
  }
}

template <typename Float>
Mesh<Float>::Mesh(size_t solve_dim, std::vector<int> dims)
    : _solve_dim{solve_dim},
      _dims(std::move(dims)), _ds{}, _dl{}, _d{}, _du{}, _dw{}, _x{}, _u{} {
  assert(_solve_dim < _dims.size() && "solve dim greater than number of dims");
  size_t num_elements =
      std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<>());
  _ds.resize(num_elements);
  _dl.resize(num_elements);
  _d.resize(num_elements);
  _du.resize(num_elements);
  _dw.resize(num_elements);
  _x.resize(num_elements);
  // _u.resize(num_elements); u stays

  size_t n_sys_in = std::accumulate(_dims.data(), _dims.data() + _solve_dim, 1,
                                    std::multiplies<>());
  size_t n_sys_out =
      std::accumulate(_dims.data() + _solve_dim + 1,
                      _dims.data() + _dims.size(), 1, std::multiplies<>());
  size_t sys_size = _dims[_solve_dim];

#pragma omp parallel
  {
    std::mt19937_64 gen(omp_get_thread_num());
    std::uniform_real_distribution<Float> dist;

#pragma omp for collapse(2)
    for (int i = 0; i < n_sys_out; ++i) {
      for (int j = 0; j < n_sys_in; ++j) {
        size_t sys_start = i * n_sys_in * sys_size + j;
        for (int n = 0; n < sys_size; ++n) {
          size_t idx = sys_start + n * n_sys_in;
          _ds[idx]   = n > 1 ? -1 * dist(gen) : 0;
          _dl[idx]   = n > 0 ? -1 * dist(gen) : 0;
          _d[idx]    = 6 * dist(gen);
          _du[idx]   = n < sys_size - 1 ? -1 * dist(gen) : 0;
          _dw[idx]   = n < sys_size - 2 ? -1 * dist(gen) : 0;
          _x[idx]    = 1 * dist(gen);
        }
      }
    }
  }
}
#endif /* ifndef PENTADSOLVER_MESH_HPP_INCLUDED */
