// forward Euler finite difference solution to the heat equation on a 2D grid

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "zfparray2.h"
#include "array2d.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define unused_(x) ((void)(x))

// constants used in the solution
class Constants {
public:
  Constants(int nx, int ny, int nt) :
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

  int nx;        // grid points in x
  int ny;        // grid points in y
  int nt;        // number of time steps (0 for default)
  int x0;        // x location of heat source
  int y0;        // y location of heat source
  double k;      // diffusion constant
  double dx;     // grid spacing in x
  double dy;     // grid spacing in y
  double dt;     // time step
  double tfinal; // minimum time to run solution to
  double pi;     // 3.141...
};

template <class array2d>
inline void
time_step_parallel(array2d& u, const Constants& c);

// advance solution in parallel via thread-safe views
template <>
inline void
time_step_parallel(zfp::array2d& u, const Constants& c)
{
#ifdef _OPENMP
  // flush shared cache to ensure cache consistency across threads
  u.flush_cache();
  // compute du/dt in parallel
  zfp::array2d du(c.nx, c.ny, u.rate(), 0, u.cache_size());
  #pragma omp parallel
  {
    // create read-only private view of entire array u
    zfp::array2d::private_const_view myu(&u);
    // create read-write private view into rectangular subset of du
    zfp::array2d::private_view mydu(&du);
    mydu.partition(omp_get_thread_num(), omp_get_num_threads());
    // process rectangular region owned by this thread
    for (uint j = 0; j < mydu.size_y(); j++) {
      int y = mydu.global_y(j);
      if (1 <= y && y <= c.ny - 2)
        for (uint i = 0; i < mydu.size_x(); i++) {
          int x = mydu.global_x(i);
          if (1 <= x && x <= c.nx - 2) {
            double uxx = (myu(x - 1, y) - 2 * myu(x, y) + myu(x + 1, y)) / (c.dx * c.dx);
            double uyy = (myu(x, y - 1) - 2 * myu(x, y) + myu(x, y + 1)) / (c.dy * c.dy);
            mydu(i, j) = c.dt * c.k * (uxx + uyy);
          }
        }
    }
    // compress all private cached blocks to shared storage
    mydu.flush_cache();
  }
  // take forward Euler step in serial
  for (uint i = 0; i < u.size(); i++)
    u[i] += du[i];
#else
  unused_(u);
  unused_(c);
#endif
}

// dummy template instantiation; never executed
template <>
inline void
time_step_parallel(raw::array2d& u, const Constants& c)
{
  unused_(u);
  unused_(c);
}

// advance solution using integer array indices
template <class array2d>
inline void
time_step_indexed(array2d& u, const Constants& c)
{
  // compute du/dt
  array2d du(c.nx, c.ny, u.rate(), 0, u.cache_size());
  for (int y = 1; y < c.ny - 1; y++) {
    for (int x = 1; x < c.nx - 1; x++) {
      double uxx = (u(x - 1, y) - 2 * u(x, y) + u(x + 1, y)) / (c.dx * c.dx);
      double uyy = (u(x, y - 1) - 2 * u(x, y) + u(x, y + 1)) / (c.dy * c.dy);
      du(x, y) = c.dt * c.k * (uxx + uyy);
    }
  }
  // take forward Euler step
  for (uint i = 0; i < u.size(); i++)
    u[i] += du[i];
}

// advance solution using array iterators
template <class array2d>
inline void
time_step_iterated(array2d& u, const Constants& c)
{
  // compute du/dt
  array2d du(c.nx, c.ny, u.rate(), 0, u.cache_size());
  for (typename array2d::iterator p = du.begin(); p != du.end(); p++) {
    int x = p.i();
    int y = p.j();
    if (1 <= x && x <= c.nx - 2 &&
        1 <= y && y <= c.ny - 2) {
      double uxx = (u(x - 1, y) - 2 * u(x, y) + u(x + 1, y)) / (c.dx * c.dx);
      double uyy = (u(x, y - 1) - 2 * u(x, y) + u(x, y + 1)) / (c.dy * c.dy);
      *p = c.dt * c.k * (uxx + uyy);
    }
  }
  // take forward Euler step
  for (typename array2d::iterator p = u.begin(), q = du.begin(); p != u.end(); p++, q++)
    *p += *q;
}

// solve heat equation using 
template <class array2d>
inline double
solve(array2d& u, const Constants& c, bool iterator, bool parallel)
{
  // initialize u with point heat source (u is assumed to be zero initialized)
  u(c.x0, c.y0) = 1;

  // iterate until final time
  double t;
  for (t = 0; t < c.tfinal; t += c.dt) {
    std::cerr << "t=" << std::setprecision(6) << std::fixed << t << std::endl;
    if (parallel)
      time_step_parallel(u, c);
    else if (iterator)
      time_step_iterated(u, c);
    else
      time_step_indexed(u, c);
  }

  return t;
}

// compute sum of array values
template <class array2d>
inline double
total(const array2d& u)
{
  double s = 0;
  const int nx = u.size_x();
  const int ny = u.size_y();
  for (int y = 1; y < ny - 1; y++)
    for (int x = 1; x < nx - 1; x++)
      s += u(x, y);
  return s;
}

// compute root mean square error with respect to exact solution
template <class array2d>
inline double
error(const array2d& u, const Constants& c, double t)
{
  double e = 0;
  for (int y = 1; y < c.ny - 1; y++) {
    double py = c.dy * (y - c.y0);
    for (int x = 1; x < c.nx - 1; x++) {
      double px = c.dx * (x - c.x0);
      double f = u(x, y);
      double g = c.dx * c.dy * std::exp(-(px * px + py * py) / (4 * c.k * t)) / (4 * c.pi * c.k * t);
      e += (f - g) * (f - g);
    }
  }
  return std::sqrt(e / ((c.nx - 2) * (c.ny - 2)));
}

inline int
usage()
{
  std::cerr << "Usage: diffusion [options]" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "-i : traverse arrays using iterators" << std::endl;
  std::cerr << "-n <nx> <ny> : number of grid points" << std::endl;
#ifdef _OPENMP
  std::cerr << "-p : use multithreading (only with compressed arrays)" << std::endl;
#endif
  std::cerr << "-t <nt> : number of time steps" << std::endl;
  std::cerr << "-r <rate> : use compressed arrays with 'rate' bits/value" << std::endl;
  std::cerr << "-c <blocks> : use 'blocks' 4x4 blocks of cache" << std::endl;
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  int nx = 100;
  int ny = 100;
  int nt = 0;
  double rate = 64;
  bool iterator = false;
  bool compression = false;
  bool parallel = false;
  int cache = 0;

  // parse command-line options
  for (int i = 1; i < argc; i++)
    if (std::string(argv[i]) == "-i")
      iterator = true;
    else if (std::string(argv[i]) == "-n") {
      if (++i == argc || sscanf(argv[i], "%i", &nx) != 1 ||
          ++i == argc || sscanf(argv[i], "%i", &ny) != 1)
        return usage();
    }
#ifdef _OPENMP
    else if (std::string(argv[i]) == "-p")
      parallel = true;
#endif
    else if (std::string(argv[i]) == "-t") {
      if (++i == argc || sscanf(argv[i], "%i", &nt) != 1)
        return usage();
    }
    else if (std::string(argv[i]) == "-r") {
      if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
        return usage();
      compression = true;
    }
    else if (std::string(argv[i]) == "-c") {
      if (++i == argc || sscanf(argv[i], "%i", &cache) != 1)
        return usage();
    }
    else
      return usage();

  if (parallel && !compression) {
    fprintf(stderr, "multithreading requires compressed arrays\n");
    return EXIT_FAILURE;
  }
  if (parallel && iterator) {
    fprintf(stderr, "multithreading does not support iterators\n");
    return EXIT_FAILURE;
  }

  Constants c(nx, ny, nt);

  double sum;
  double err;
  if (compression) {
    // solve problem using compressed arrays
    zfp::array2d u(nx, ny, rate, 0, cache * 4 * 4 * sizeof(double));
    rate = u.rate();
    double t = solve(u, c, iterator, parallel);
    sum = total(u);
    err = error(u, c, t);
  }
  else {
    // solve problem using uncompressed arrays
    raw::array2d u(nx, ny);
    double t = solve(u, c, iterator, parallel);
    sum = total(u);
    err = error(u, c, t);
  }

  std::cerr.unsetf(std::ios::fixed);
  std::cerr << "rate=" << rate << " sum=" << std::fixed << sum << " error=" << std::setprecision(6) << std::scientific << err << std::endl;

  return 0;
}
