// forward Euler finite difference solution to the heat equation on a 2D grid

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "zfp/array2.hpp"
#include "zfp/constarray2.hpp"
#include "zfp/codec/gencodec.hpp"
#include "array2d.hpp"

// add half precision if compiler supports it
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#include <cfloat>
#ifdef FLT16_MAX
  #define WITH_HALF 1
#else
  #undef WITH_HALF
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// uncompressed tiled arrays based on zfp generic codec
namespace tiled {
#if WITH_HALF
  typedef zfp::array2< double, zfp::codec::generic2<double, _Float16> > array2h;
#endif
  typedef zfp::array2< double, zfp::codec::generic2<double, float> > array2f;
  typedef zfp::array2< double, zfp::codec::generic2<double, double> > array2d;
}

// enumeration of uncompressed storage types
enum storage_type {
  type_none = 0,
  type_half = 1,
  type_float = 2,
  type_double = 3
};

// constants used in the solution
class Constants {
public:
  Constants(size_t nx, size_t ny, size_t nt) :
    nx(nx),
    ny(ny),
    nt(nt),
    x0((nx - 1) / 2),
    y0((ny - 1) / 2),
    k(0.04),
    dx(2.0 / (std::max(nx, ny) - 1)),
    dy(2.0 / (std::max(nx, ny) - 1)),
    dt(0.5 * (dx * dx + dy * dy) / (8 * k)),
    tfinal(nt ? nt * dt : 1.0),
    pi(3.14159265358979323846)
  {}

  size_t nx;     // grid points in x
  size_t ny;     // grid points in y
  size_t nt;     // number of time steps (0 for default)
  size_t x0;     // x location of heat source
  size_t y0;     // y location of heat source
  double k;      // diffusion constant
  double dx;     // grid spacing in x
  double dy;     // grid spacing in y
  double dt;     // time step
  double tfinal; // minimum time to run solution to
  double pi;     // 3.141...
};

// compute Laplacian uxx + uyy at (x, y)
template <class array2d>
inline double
laplacian(const array2d& u, size_t x, size_t y, const Constants& c)
{
  double uxx = (u(x - 1, y) - 2 * u(x, y) + u(x + 1, y)) / (c.dx * c.dx);
  double uyy = (u(x, y - 1) - 2 * u(x, y) + u(x, y + 1)) / (c.dy * c.dy);
  return uxx + uyy;
}

template <class state, class scratch>
inline void
time_step_parallel(state& u, scratch& v, const Constants& c);

#ifdef _OPENMP
// advance solution in parallel via thread-safe views
template <>
inline void
time_step_parallel(zfp::array2d& u, zfp::array2d& du, const Constants& c)
{
  // flush shared cache to ensure cache consistency across threads
  u.flush_cache();
  // zero-initialize du
  du.set(0);
  // compute du/dt in parallel
  #pragma omp parallel
  {
    // create read-only private view of entire array u
    zfp::array2d::private_const_view myu(&u);
    // create read-write private view into rectangular subset of du
    zfp::array2d::private_view mydu(&du);
    mydu.partition(omp_get_thread_num(), omp_get_num_threads());
    // process rectangular region owned by this thread
    for (size_t j = 0; j < mydu.size_y(); j++) {
      size_t y = mydu.global_y(j);
      if (1 <= y && y <= c.ny - 2)
        for (size_t i = 0; i < mydu.size_x(); i++) {
          size_t x = mydu.global_x(i);
          if (1 <= x && x <= c.nx - 2)
            mydu(i, j) = c.dt * c.k * laplacian(myu, x, y, c);
        }
    }
    // compress all private cached blocks to shared storage
    mydu.flush_cache();
  }
  // take forward Euler step in serial
  for (size_t i = 0; i < u.size(); i++)
    u[i] += du[i];
}
#else
// dummy template instantiation when OpenMP support is not available
template <>
inline void time_step_parallel(zfp::array2d&, zfp::array2d&, const Constants&) {}
#endif

// dummy template instantiations; never executed
template <>
inline void time_step_parallel(zfp::const_array2d&, raw::array2d&, const Constants&) {}
template <>
inline void time_step_parallel(raw::array2d&, raw::array2d&, const Constants&) {}
template <>
inline void time_step_parallel(tiled::array2d&, tiled::array2d&, const Constants&) {}
template <>
inline void time_step_parallel(tiled::array2f&, tiled::array2f&, const Constants&) {}
#if WITH_HALF
template <>
inline void time_step_parallel(tiled::array2h&, tiled::array2h&, const Constants&) {}
#endif

// advance solution using integer array indices (generic implementation)
template <class state, class scratch>
inline void
time_step_indexed(state& u, scratch& du, const Constants& c)
{
  // compute du/dt
  for (size_t y = 1; y < c.ny - 1; y++)
    for (size_t x = 1; x < c.nx - 1; x++)
      du(x, y) = c.dt * c.k * laplacian(u, x, y, c);
  // take forward Euler step
  for (uint i = 0; i < u.size(); i++)
    u[i] += du[i];
}

// advance solution using integer array indices (read-only arrays)
template <>
inline void
time_step_indexed(zfp::const_array2d& u, raw::array2d& v, const Constants& c)
{
  // initialize v as uncompressed copy of u
  u.get(&v[0]);
  // take forward Euler step v += (du/dt) dt
  for (size_t y = 1; y < c.ny - 1; y++)
    for (size_t x = 1; x < c.nx - 1; x++)
      v(x, y) += c.dt * c.k * laplacian(u, x, y, c);
  // update u with uncompressed copy v
  u.set(&v[0]);
}

// advance solution using array iterators (generic implementation)
template <class state, class scratch>
inline void
time_step_iterated(state& u, scratch& du, const Constants& c)
{
  // compute du/dt
  for (typename scratch::iterator q = du.begin(); q != du.end(); q++) {
    size_t x = q.i();
    size_t y = q.j();
    if (1 <= x && x <= c.nx - 2 &&
        1 <= y && y <= c.ny - 2)
      *q = c.dt * c.k * laplacian(u, x, y, c);
  }
  // take forward Euler step
  for (typename state::iterator p = u.begin(); p != u.end(); p++)
    *p += du(p.i(), p.j());
}

// advance solution using array iterators (read-only arrays)
template <>
inline void
time_step_iterated(zfp::const_array2d& u, raw::array2d& v, const Constants& c)
{
  // initialize v as uncompressed copy of u
  u.get(&v[0]);
  // take forward Euler step v += (du/dt) dt
  for (raw::array2d::iterator q = v.begin(); q != v.end(); q++) {
    size_t x = q.i();
    size_t y = q.j();
    if (1 <= x && x <= c.nx - 2 &&
        1 <= y && y <= c.ny - 2)
      *q += c.dt * c.k * laplacian(u, x, y, c);
  }
  // update u with uncompressed copy v
  u.set(&v[0]);
}

// set initial conditions with a point heat source (u is assumed zero-initialized)
template <class state, class scratch>
inline void
initialize(state& u, scratch&, const Constants& c)
{
  u(c.x0, c.y0) = 1;
}

// set initial conditions for const_array; requires updating the whole array
template <>
inline void
initialize(zfp::const_array2d& u, raw::array2d& v, const Constants& c)
{
  v(c.x0, c.y0) = 1;
  u.set(&v[0]);
}

// solve heat equation
template <class state, class scratch>
inline double
solve(state& u, scratch& v, const Constants& c, bool iterator, bool parallel)
{
  // initialize u with point heat source
  initialize(u, v, c);

  // iterate until final time
  double t;
  for (t = 0; t < c.tfinal; t += c.dt) {
    // print time and effective rate
    double rate = double(u.size_bytes(ZFP_DATA_PAYLOAD)) * CHAR_BIT / u.size();
    double rest = double(u.size_bytes(ZFP_DATA_ALL ^ ZFP_DATA_PAYLOAD) * CHAR_BIT / u.size());
    std::cerr << "time=" << std::setprecision(6) << std::fixed << t << " ";
    std::cerr << "rate=" << std::setprecision(3) << std::fixed << rate << " (+" << rest << ")" << std::endl;
    // advance solution one time step
    if (parallel)
      time_step_parallel(u, v, c);
    else if (iterator)
      time_step_iterated(u, v, c);
    else
      time_step_indexed(u, v, c);
  }

  return t;
}

// compute sum of array values
template <class state>
inline double
total(const state& u)
{
  double s = 0;
  const size_t nx = u.size_x();
  const size_t ny = u.size_y();
  for (size_t y = 1; y < ny - 1; y++)
    for (size_t x = 1; x < nx - 1; x++)
      s += u(x, y);
  return s;
}

// compute root mean square error with respect to exact solution
template <class state>
inline double
error(const state& u, const Constants& c, double t)
{
  double e = 0;
  for (size_t y = 1; y < c.ny - 1; y++) {
    double py = c.dy * ((int)y - (int)c.y0);
    for (size_t x = 1; x < c.nx - 1; x++) {
      double px = c.dx * ((int)x - (int)c.x0);
      double f = u(x, y);
      double g = c.dx * c.dy * std::exp(-(px * px + py * py) / (4 * c.k * t)) / (4 * c.pi * c.k * t);
      e += (f - g) * (f - g);
    }
  }
  return std::sqrt(e / ((c.nx - 2) * (c.ny - 2)));
}

// execute solver and evaluate error
template <class state, class scratch>
inline void
execute(state& u, scratch& v, size_t nt, bool iterator, bool parallel)
{
  Constants c(u.size_x(), u.size_y(), nt);
  double t = solve(u, v, c, iterator, parallel);
  double sum = total(u);
  double err = error(u, c, t);
  std::cerr.unsetf(std::ios::fixed);
  std::cerr << "sum=" << std::setprecision(6) << std::fixed << sum << " error=" << std::setprecision(6) << std::scientific << err << std::endl;
}

// print usage information
inline int
usage()
{
  std::cerr << "Usage: diffusion [options]" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "-a <tolerance> : use compressed arrays with given absolute error tolerance" << std::endl;
  std::cerr << "-b <blocks> : use 'blocks' 4x4 blocks of cache" << std::endl;
  std::cerr << "-c : use read-only compressed arrays" << std::endl;
  std::cerr << "-d : use double-precision tiled arrays" << std::endl;
  std::cerr << "-f : use single-precision tiled arrays" << std::endl;
#if WITH_HALF
  std::cerr << "-h : use half-precision tiled arrays" << std::endl;
#endif
  std::cerr << "-i : traverse arrays using iterators" << std::endl;
#ifdef _OPENMP
  std::cerr << "-j : use multithreading (only with compressed arrays)" << std::endl;
#endif
  std::cerr << "-n <nx> <ny> : number of grid points" << std::endl;
  std::cerr << "-p <precision> : use compressed arrays with given precision" << std::endl;
  std::cerr << "-r <rate> : use compressed arrays with given compressed bits/value" << std::endl;
  std::cerr << "-R : use compressed arrays with lossless compression" << std::endl;
  std::cerr << "-t <nt> : number of time steps" << std::endl;
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  size_t nx = 128;
  size_t ny = 128;
  size_t nt = 0;
  size_t cache_size = 0;
  zfp_config config = zfp_config_none();
  bool iterator = false;
  bool parallel = false;
  bool writable = true;
  storage_type type = type_none;

  // parse command-line options
  for (int i = 1; i < argc; i++)
    if (std::string(argv[i]) == "-a") {
      double tolerance;
      if (++i == argc || sscanf(argv[i], "%lf", &tolerance) != 1)
        return usage();
      config = zfp_config_accuracy(tolerance);
    }
    else if (std::string(argv[i]) == "-b") {
      if (++i == argc || (std::istringstream(argv[i]) >> cache_size).fail())
        return usage();
      cache_size *= 4 * 4 * sizeof(double);
    }
    else if (std::string(argv[i]) == "-c")
      writable = false;
    else if (std::string(argv[i]) == "-d")
      type = type_double;
    else if (std::string(argv[i]) == "-f")
      type = type_float;
#if WITH_HALF
    else if (std::string(argv[i]) == "-h")
      type = type_half;
#endif
    else if (std::string(argv[i]) == "-i")
      iterator = true;
#ifdef _OPENMP
    else if (std::string(argv[i]) == "-j")
      parallel = true;
#endif
    else if (std::string(argv[i]) == "-n") {
      if (++i == argc || (std::istringstream(argv[i]) >> nx).fail() ||
          ++i == argc || (std::istringstream(argv[i]) >> ny).fail())
        return usage();
    }
    else if (std::string(argv[i]) == "-p") {
      uint precision;
      if (++i == argc || sscanf(argv[i], "%u", &precision) != 1)
        return usage();
      config = zfp_config_precision(precision);
    }
    else if (std::string(argv[i]) == "-r") {
      double rate;
      if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
        return usage();
      config = zfp_config_rate(rate, false);
    }
    else if (std::string(argv[i]) == "-R")
      config = zfp_config_reversible();
    else if (std::string(argv[i]) == "-t") {
      if (++i == argc || (std::istringstream(argv[i]) >> nt).fail())
        return usage();
    }
    else
      return usage();

  bool compression = (config.mode != zfp_mode_null);

  // sanity check command-line arguments
  if (parallel && !compression) {
    fprintf(stderr, "multithreading requires compressed arrays\n");
    return EXIT_FAILURE;
  }
  if (parallel && !writable) {
    fprintf(stderr, "multithreading requires read-write arrays\n");
    return EXIT_FAILURE;
  }
  if (parallel && iterator) {
    fprintf(stderr, "multithreading does not support iterators\n");
    return EXIT_FAILURE;
  }
  if (compression && writable && config.mode != zfp_mode_fixed_rate) {
    fprintf(stderr, "compression mode requires read-only arrays (-c)\n");
    return EXIT_FAILURE;
  }
  if (!writable && !compression) {
    fprintf(stderr, "read-only arrays require compression parameters\n");
    return EXIT_FAILURE;
  }
  if (compression && type != type_none) {
    fprintf(stderr, "tiled arrays do not support compression parameters\n");
    return EXIT_FAILURE;
  }

  // if unspecified, set cache size to two layers of blocks
  if (!cache_size)
    cache_size = 2 * 4 * nx * sizeof(double);

  // solve problem
  if (compression) {
    // use compressed arrays
    if (writable) {
      // use read-write fixed-rate arrays
      zfp::array2d u(nx, ny, config.arg.rate, 0, cache_size);
      zfp::array2d v(nx, ny, config.arg.rate, 0, cache_size);
      execute(u, v, nt, iterator, parallel);
    }
    else {
      // use read-only variable-rate arrays
      zfp::const_array2d u(nx, ny, config, 0, cache_size);
      raw::array2d v(nx, ny);
      execute(u, v, nt, iterator, parallel);
    }
  }
  else {
    // use uncompressed arrays
    switch (type) {
#if WITH_HALF
      case type_half: {
          // use zfp generic codec with tiled half-precision storage
          tiled::array2h u(nx, ny, sizeof(__fp16) * CHAR_BIT, 0, cache_size);
          tiled::array2h v(nx, ny, sizeof(__fp16) * CHAR_BIT, 0, cache_size);
          execute(u, v, nt, iterator, parallel);
        }
        break;
#endif
      case type_float: {
          // use zfp generic codec with tiled single-precision storage
          tiled::array2f u(nx, ny, sizeof(float) * CHAR_BIT, 0, cache_size);
          tiled::array2f v(nx, ny, sizeof(float) * CHAR_BIT, 0, cache_size);
          execute(u, v, nt, iterator, parallel);
        }
        break;
      case type_double: {
          // use zfp generic codec with tiled double-precision storage
          tiled::array2d u(nx, ny, sizeof(double) * CHAR_BIT, 0, cache_size);
          tiled::array2d v(nx, ny, sizeof(double) * CHAR_BIT, 0, cache_size);
          execute(u, v, nt, iterator, parallel);
        }
        break;
      default: {
          // use uncompressed array with row-major double-precision storage
          raw::array2d u(nx, ny, sizeof(double) * CHAR_BIT);
          raw::array2d v(nx, ny, sizeof(double) * CHAR_BIT);
          execute(u, v, nt, iterator, parallel);
        }
        break;
    }
  }

  return 0;
}
