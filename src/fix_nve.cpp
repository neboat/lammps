// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_nve.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"

// #define forall for

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixNVE::FixNVE(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (!utils::strmatch(style,"^nve/sphere") && narg < 3)
    utils::missing_cmd_args(FLERR, fmt::format("fix {}", style), error);

  auto plain_style = utils::strip_style_suffix(style, lmp);
  if (utils::strmatch(plain_style, "^nve$") && narg > 3)
    error->all(FLERR, 3, "Unsupported additional arguments for fix {}", style);

  dynamic_group_allow = 1;
  time_integrate = 1;
  fuse_integrate_flag = 1;
}

/* ---------------------------------------------------------------------- */

int FixNVE::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNVE::init()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;

  if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixNVE::initial_integrate(int /*vflag*/)
{
  // double dtfm;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double *_x = *x;
  double *_v = *v;
  double *_f = *f;
  double _dtf = dtf;
  double _dtv = dtv;
  if (rmass) {
    forall (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = _dtf / rmass[i];
        _v[3*i+0] += dtfm * _f[3*i+0];
        _v[3*i+1] += dtfm * _f[3*i+1];
        _v[3*i+2] += dtfm * _f[3*i+2];
        _x[3*i+0] += _dtv * _v[3*i+0];
        _x[3*i+1] += _dtv * _v[3*i+1];
        _x[3*i+2] += _dtv * _v[3*i+2];
      }

  } else {
    forall (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = _dtf / mass[type[i]];
        _v[3*i+0] += dtfm * _f[3*i+0];
        _v[3*i+1] += dtfm * _f[3*i+1];
        _v[3*i+2] += dtfm * _f[3*i+2];
        _x[3*i+0] += _dtv * _v[3*i+0];
        _x[3*i+1] += _dtv * _v[3*i+1];
        _x[3*i+2] += _dtv * _v[3*i+2];
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate()
{
  // double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double *_v = *v;
  double *_f = *f;
  double _dtf = dtf;
  if (rmass) {
    forall (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = _dtf / rmass[i];
        _v[3*i+0] += dtfm * _f[3*i+0];
        _v[3*i+1] += dtfm * _f[3*i+1];
        _v[3*i+2] += dtfm * _f[3*i+2];
      }

  } else {
    forall (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = _dtf / mass[type[i]];
        _v[3*i+0] += dtfm * _f[3*i+0];
        _v[3*i+1] += dtfm * _f[3*i+1];
        _v[3*i+2] += dtfm * _f[3*i+2];
      }
  }
}

static void fused_integrate_rmass_loop(double *__restrict__* x, double *__restrict__* v, double *__restrict__* f,
                                       double *__restrict__ rmass, int *__restrict__ mask, int groupbit,
                                       double dtf, double dtv, int nlocal)
{
  double *_x = *x;
  double *_v = *v;
  double *_f = *f;
  forall (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      const double dtfm = 2.0 * dtf / rmass[i];
      _v[3*i+0] += dtfm * _f[3*i+0];
      _v[3*i+1] += dtfm * _f[3*i+1];
      _v[3*i+2] += dtfm * _f[3*i+2];
      _x[3*i+0] += dtv * _v[3*i+0];
      _x[3*i+1] += dtv * _v[3*i+1];
      _x[3*i+2] += dtv * _v[3*i+2];
    }
}

static void fused_integrate_loop(double **__restrict__ x, double **__restrict__ v, double **__restrict__ f,
                                 double *__restrict__ mass, int *__restrict__ type, int *__restrict__ mask,
                                 int groupbit, double dtf, double dtv, int nlocal)
{
  double *_x = *x;
  double *_v = *v;
  double *_f = *f;
  forall (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      const double dtfm = 2.0 * dtf / mass[type[i]];
      _v[3*i+0] += dtfm * _f[3*i+0];
      _v[3*i+1] += dtfm * _f[3*i+1];
      _v[3*i+2] += dtfm * _f[3*i+2];
      _x[3*i+0] += dtv * _v[3*i+0];
      _x[3*i+1] += dtv * _v[3*i+1];
      _x[3*i+2] += dtv * _v[3*i+2];
    }
}

void FixNVE::fused_integrate(int)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    fused_integrate_rmass_loop(x, v, f, rmass, mask, groupbit, dtf, dtv, nlocal);
  } else {
    fused_integrate_loop(x, v, f, mass, type, mask, groupbit, dtf, dtv, nlocal);
  }
}

/* ---------------------------------------------------------------------- */

void FixNVE::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
