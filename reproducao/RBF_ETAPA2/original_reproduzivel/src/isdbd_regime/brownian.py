from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .profiles import Geometry, ProfileCoefficients

@dataclass(frozen=True)
class BrownianFields:
    g: np.ndarray
    ell: np.ndarray
    D_nm2_ns: np.ndarray
    g_prime_per_nm: np.ndarray
    ell_prime_per_nm: np.ndarray
    D_prime_nm_ns: np.ndarray
    ito_drift_nm_ns: np.ndarray

def _arr(x):
    z=np.asarray(x,dtype=np.float64)
    if z.ndim!=1 or len(z)<1 or not np.all(np.isfinite(z)):
        raise ValueError("positions must be a finite 1D array")
    return z

def _qc(z,s):
    return np.exp(-(z*z)/(2.0*s*s))

def _dqc(z,s):
    return -(z/(s*s))*_qc(z,s)

def _qp(z,m,s):
    return np.exp(-((z-m)**2)/(2.0*s*s))+np.exp(-((z+m)**2)/(2.0*s*s))

def _dqp(z,m,s):
    a=np.exp(-((z-m)**2)/(2.0*s*s))
    b=np.exp(-((z+m)**2)/(2.0*s*s))
    return -((z-m)/(s*s))*a-((z+m)/(s*s))*b

def _anchored_fields(z,geom,coeff):
    geom.validate()
    a=coeff.as_array()
    H=geom.H_nm

    q0=_qc(z,geom.sigma_center_nm)-_qc(np.array([H]),geom.sigma_center_nm)[0]
    q1=_qp(z,geom.mu_inner_nm,geom.sigma_inner_nm)-_qp(np.array([H]),geom.mu_inner_nm,geom.sigma_inner_nm)[0]
    q2=_qp(z,geom.mu_outer_nm,geom.sigma_outer_nm)-_qp(np.array([H]),geom.mu_outer_nm,geom.sigma_outer_nm)[0]

    d0=_dqc(z,geom.sigma_center_nm)
    d1=_dqp(z,geom.mu_inner_nm,geom.sigma_inner_nm)
    d2=_dqp(z,geom.mu_outer_nm,geom.sigma_outer_nm)

    val=a[0]*q0+a[1]*q1+a[2]*q2
    der=a[0]*d0+a[1]*d1+a[2]*d2
    return val,der

def analytic_fields(z_nm,g_geometry,d_geometry,a,b,D_bulk_nm2_ns):
    z=_arr(z_nm)
    g_geometry.validate()
    d_geometry.validate()
    if g_geometry.H_nm!=d_geometry.H_nm:
        raise ValueError("H mismatch")
    H=g_geometry.H_nm
    if np.any(z < -H) or np.any(z > H):
        raise ValueError("positions outside domain")
    if not np.isfinite(D_bulk_nm2_ns) or D_bulk_nm2_ns<=0:
        raise ValueError("D_bulk must be positive finite")

    g,gp=_anchored_fields(z,g_geometry,a)
    ell,ellp=_anchored_fields(z,d_geometry,b)
    D=D_bulk_nm2_ns*np.exp(ell)
    Dp=D*ellp
    drift=D*(ellp-gp)

    if not all(np.all(np.isfinite(x)) for x in (g,ell,D,gp,ellp,Dp,drift)):
        raise FloatingPointError("non-finite analytic field")
    if np.any(D<=0):
        raise FloatingPointError("non-positive diffusivity")

    return BrownianFields(g,ell,D,gp,ellp,Dp,drift)

def reflect_interval(x_nm,H_nm):
    x=np.asarray(x_nm,dtype=np.float64)
    if not np.isfinite(H_nm) or H_nm<=0:
        raise ValueError("H must be positive finite")
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite positions")
    period=4.0*H_nm
    y=np.mod(x+H_nm,period)
    return np.where(y<=2.0*H_nm,-H_nm+y,3.0*H_nm-y)

def em_step(z_nm,dt_ns,rng,g_geometry,d_geometry,a,b,D_bulk_nm2_ns):
    if not np.isfinite(dt_ns) or dt_ns<=0:
        raise ValueError("dt must be positive finite")
    z=_arr(z_nm)
    f=analytic_fields(z,g_geometry,d_geometry,a,b,D_bulk_nm2_ns)
    xi=rng.standard_normal(len(z))
    trial=z+f.ito_drift_nm_ns*dt_ns+np.sqrt(2.0*f.D_nm2_ns*dt_ns)*xi
    return reflect_interval(trial,g_geometry.H_nm)

def milstein_step(z_nm,dt_ns,rng,g_geometry,d_geometry,a,b,D_bulk_nm2_ns):
    if not np.isfinite(dt_ns) or dt_ns<=0:
        raise ValueError("dt must be positive finite")
    z=_arr(z_nm)
    f=analytic_fields(z,g_geometry,d_geometry,a,b,D_bulk_nm2_ns)
    xi=rng.standard_normal(len(z))
    trial=(
        z
        + f.ito_drift_nm_ns*dt_ns
        + np.sqrt(2.0*f.D_nm2_ns*dt_ns)*xi
        + 0.5*f.D_prime_nm_ns*dt_ns*(xi*xi-1.0)
    )
    return reflect_interval(trial,g_geometry.H_nm)

def simulate_reflecting(
    z0_nm,dt_ns,n_steps,seed,method,
    g_geometry,d_geometry,a,b,D_bulk_nm2_ns
):
    z=_arr(z0_nm).copy()
    if int(n_steps)!=n_steps or n_steps<0:
        raise ValueError("n_steps must be a nonnegative integer")
    method=str(method).upper()
    if method not in {"EM","MILSTEIN"}:
        raise ValueError("method must be EM or MILSTEIN")
    rng=np.random.default_rng(int(seed))
    step=em_step if method=="EM" else milstein_step
    for _ in range(int(n_steps)):
        z=step(z,dt_ns,rng,g_geometry,d_geometry,a,b,D_bulk_nm2_ns)
    return z
