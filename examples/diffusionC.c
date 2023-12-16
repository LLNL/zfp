/*
forward Euler finite difference solution to the heat equation on a 2D grid
(ported to C, from diffusion.cpp)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "zfp/array.h"

#define _ (CFP_NAMESPACE.array2d)

#define MAX(x, y) (((nx) > (ny)) ? (nx) : (ny))

/* constants used in the solution */
typedef struct {
  size_t nx;     /* grid points in x */
  size_t ny;     /* grid points in y */
  int nt;        /* number of time steps (0 for default) */
  int x0;        /* x location of heat source */
  int y0;        /* y location of heat source */
  double k;      /* diffusion constant */
  double dx;     /* grid spacing in x */
  double dy;     /* grid spacing in y */
  double dt;     /* time step */
  double tfinal; /* minimum time to run solution to */
  double pi;     /* 3.141... */
} constants;

void
init_constants(constants* c, int nx, int ny, int nt)
{
  c->nx = nx;
  c->ny = ny;
  c->nt = nt;
  c->x0 = (nx - 1) / 2;
  c->y0 = (ny - 1) / 2;
  c->k = 0.04;
  c->dx = 2.0 / (MAX(nx, ny) - 1);
  c->dy = 2.0 / (MAX(nx, ny) - 1);
  c->dt = 0.5 * (c->dx * c->dx + c->dy * c->dy) / (8 * c->k);
  c->tfinal = nt ? nt * c->dt : 1.0;
  c->pi = 3.14159265358979323846;
}

/* advance solution using integer array indices */
static void
time_step_indexed_compressed(cfp_array2d u, const constants* c)
{
  /* compute du/dt */
  cfp_array2d du = _.ctor(c->nx, c->ny, _.rate(u), 0, _.cache_size(u));
  size_t i, x, y;
  for (y = 1; y < c->ny - 1; y++) {
    for (x = 1; x < c->nx - 1; x++) {
      double uxx = (_.get(u, x - 1, y) - 2 * _.get(u, x, y) + _.get(u, x + 1, y)) / (c->dx * c->dx);
      double uyy = (_.get(u, x, y - 1) - 2 * _.get(u, x, y) + _.get(u, x, y + 1)) / (c->dy * c->dy);
      _.set(du, x, y, c->dt * c->k * (uxx + uyy));
    }
  }
  /* take forward Euler step */
  for (i = 0; i < _.size(u); i++) {
    /* u[i] += du[i] */
    double val = _.get_flat(u, i) + _.get_flat(du, i);
    _.set_flat(u, i, val);
  }

  _.dtor(du);
}

/* advance solution using array iterators */
static void
time_step_iterated_compressed(cfp_array2d u, const constants* c)
{
  /* compute du/dt */
  cfp_array2d du = _.ctor(c->nx, c->ny, _.rate(u), 0, _.cache_size(u));
  cfp_iter2d p, q;
  for (q = _.begin(du); _.iterator.neq(q, _.end(du)); q = _.iterator.inc(q)) {
    size_t x = _.iterator.i(q);
    size_t y = _.iterator.j(q);
    if (1 <= x && x <= c->nx - 2 &&
        1 <= y && y <= c->ny - 2) {
      double uxx = (_.get(u, x - 1, y) - 2 * _.get(u, x, y) + _.get(u, x + 1, y)) / (c->dx * c->dx);
      double uyy = (_.get(u, x, y - 1) - 2 * _.get(u, x, y) + _.get(u, x, y + 1)) / (c->dy * c->dy);
      _.iterator.set(q, c->dt * c->k * (uxx + uyy));
    }
  }
  /* take forward Euler step */
  for (p = _.begin(u), q = _.begin(du); _.iterator.neq(p, _.end(u)); p = _.iterator.inc(p), q = _.iterator.inc(q)) {
    /* u[i] += du[i] */
    double val = _.iterator.get(p) + _.iterator.get(q);
    _.iterator.set(p, val);
  }

  _.dtor(du);
}

/* advance solution using integer array indices */
static void
time_step_indexed(double* u, const constants* c)
{
  /* compute du/dt */
  double* du = calloc(c->nx * c->ny, sizeof(double));
  size_t i, x, y;
  for (y = 1; y < c->ny - 1; y++)
    for (x = 1; x < c->nx - 1; x++) {
      double uxx = (u[(x - 1) + c->nx * y] - 2 * u[x + c->nx * y] + u[(x + 1) + c->nx * y]) / (c->dx * c->dx);
      double uyy = (u[x + c->nx * (y - 1)] - 2 * u[x + c->nx * y] + u[x + c->nx * (y + 1)]) / (c->dy * c->dy);
      du[x + c->nx * y] = c->dt * c->k * (uxx + uyy);
    }
  /* take forward Euler step */
  for (i = 0; i < c->nx * c->ny; i++)
    u[i] += du[i];

  free(du);
}

/* solve heat equation using compressed arrays */
static double
solve_compressed(cfp_array2d u, const constants* c, zfp_bool iterator)
{
  double t;

  /* initialize u with point heat source (u is assumed to be zero initialized) */
  _.set(u, c->x0, c->y0, 1);

  /* iterate until final time */
  for (t = 0; t < c->tfinal; t += c->dt) {
    fprintf(stderr, "t=%lf\n", t);
    if (iterator)
      time_step_iterated_compressed(u, c);
    else
      time_step_indexed_compressed(u, c);
  }

  return t;
}

/* solve heat equation using uncompressed arrays */
static double
solve(double* u, const constants* c)
{
  double t;

  /* initialize u with point heat source (u is assumed to be zero initialized) */
  u[c->x0 + c->nx * c->y0] = 1;

  /* iterate until final time */
  for (t = 0; t < c->tfinal; t += c->dt) {
    fprintf(stderr, "t=%lf\n", t);
    time_step_indexed(u, c);
  }

  return t;
}

/* compute sum of array values */
static double
total_compressed(const cfp_array2d u)
{
  double s = 0;
  const size_t nx = _.size_x(u);
  const size_t ny = _.size_y(u);
  size_t x, y;
  for (y = 1; y < ny - 1; y++)
    for (x = 1; x < nx - 1; x++)
      s += _.get(u, x, y);
  return s;
}

/* compute sum of array values */
static double
total(const double* u, size_t nx, size_t ny)
{
  double s = 0;
  size_t x, y;
  for (y = 1; y < ny - 1; y++)
    for (x = 1; x < nx - 1; x++)
      s += u[x + nx * y];
  return s;
}

/* compute root mean square error with respect to exact solution */
static double
error_compressed(const cfp_array2d u, const constants* c, double t)
{
  double e = 0;
  size_t x, y;
  for (y = 1; y < c->ny - 1; y++) {
    double py = c->dy * ((int)y - (int)c->y0);
    for (x = 1; x < c->nx - 1; x++) {
      double px = c->dx * ((int)x - (int)c->x0);
      double f = _.get(u, x, y);
      double g = c->dx * c->dy * exp(-(px * px + py * py) / (4 * c->k * t)) / (4 * c->pi * c->k * t);
      e += (f - g) * (f - g);
    }
  }
  return sqrt(e / ((c->nx - 2) * (c->ny - 2)));
}

/* compute root mean square error with respect to exact solution */
static double
error(const double* u, const constants* c, double t)
{
  double e = 0;
  size_t x, y;
  for (y = 1; y < c->ny - 1; y++) {
    double py = c->dy * ((int)y - (int)c->y0);
    for (x = 1; x < c->nx - 1; x++) {
      double px = c->dx * ((int)x - (int)c->x0);
      double f = u[x + c->nx * y];
      double g = c->dx * c->dy * exp(-(px * px + py * py) / (4 * c->k * t)) / (4 * c->pi * c->k * t);
      e += (f - g) * (f - g);
    }
  }
  return sqrt(e / ((c->nx - 2) * (c->ny - 2)));
}

static int
usage(void)
{
  fprintf(stderr, "Usage: diffusionC [options]\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "-b <blocks> : use 'blocks' 4x4 blocks of cache\n");
  fprintf(stderr, "-i : traverse arrays using iterators\n");
  fprintf(stderr, "-n <nx> <ny> : number of grid points\n");
  fprintf(stderr, "-r <rate> : use compressed arrays with given compressed bits/value\n");
  fprintf(stderr, "-t <nt> : number of time steps\n");
  return EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
  int nx = 128;
  int ny = 128;
  int nt = 0;
  int cache_size = 0;
  double rate = 64;
  zfp_bool iterator = zfp_false;
  zfp_bool compression = zfp_false;
  constants* c = 0;
  double sum;
  double err;

  /* parse command-line options */
  int i;
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-' || argv[i][2])
      return usage();
    switch(argv[i][1]) {
      case 'b':
        if (++i == argc || sscanf(argv[i], "%d", &cache_size) != 1)
          return usage();
        cache_size *= (int)(4 * 4 * sizeof(double));
        break;
      case 'i':
        iterator = zfp_true;
        break;
      case 'n':
        if (++i == argc || sscanf(argv[i], "%d", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%d", &ny) != 1)
          return usage();
        break;
      case 'r':
        if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
          return usage();
        compression = zfp_true;
        break;
      case 't':
        if (++i == argc || sscanf(argv[i], "%d", &nt) != 1)
          return usage();
        break;
      default:
        return usage();
    }
  }

  c = malloc(sizeof(constants));
  init_constants(c, nx, ny, nt);

  if (compression) {
    /* solve problem using compressed arrays */
    cfp_array2d u = _.ctor(nx, ny, rate, 0, cache_size);
    double t = solve_compressed(u, c, iterator);
    sum = total_compressed(u);
    err = error_compressed(u, c, t);
    rate = _.rate(u);
    _.dtor(u);
  }
  else {
    /* solve problem using primitive arrays */
    double* u = calloc(nx * ny, sizeof(double));
    double t = solve(u, c);
    sum = total(u, nx, ny);
    err = error(u, c, t);
    free(u);
  }

  fprintf(stderr, "rate=%g sum=%g error=%.6e\n", rate, sum, err);

  free(c);

  return 0;
}
