from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded

@dataclass(frozen=True)
class FPResult:
    p: np.ndarray
    interface_flux: np.ndarray
    flux_mean: float
    permeability_nm_per_ns: float
    permeability_cm_per_s: float
    relative_flux_nonuniformity: float
    method: str="SCHAFETTER_GUMMEL_FINITE_VOLUME"

def bernoulli(x):
    x=np.asarray(x,dtype=np.float64)
    out=np.empty_like(x)
    small=np.abs(x)<1.0e-6
    y=x[small]
    out[small]=1.0-y/2.0+y*y/12.0-y**4/720.0
    out[~small]=x[~small]/np.expm1(x[~small])
    return out

def _validate(z,g,D):
    z=np.asarray(z,dtype=np.float64)
    g=np.asarray(g,dtype=np.float64)
    D=np.asarray(D,dtype=np.float64)
    if z.ndim!=1 or len(z)<3 or g.shape!=z.shape or D.shape!=z.shape:
        raise ValueError("z, g, D must be matching 1D arrays with >=3 points")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(g)) and np.all(np.isfinite(D))):
        raise ValueError("non-finite input")
    if np.any(D<=0): raise ValueError("D must be positive")
    dz=np.diff(z)
    if np.any(dz<=0): raise ValueError("z must increase")
    h=float(dz[0])
    if not np.allclose(dz,h,rtol=2e-13,atol=2e-15):
        raise ValueError("uniform grid required")
    return z,g,D,h

def interface_flux_sg(z,g,D,p):
    z,g,D,h=_validate(z,g,D)
    p=np.asarray(p,dtype=np.float64)
    if p.shape!=z.shape or not np.all(np.isfinite(p)):
        raise ValueError("invalid p")
    dg=np.diff(g)
    Df=np.sqrt(D[:-1]*D[1:])
    Bp=bernoulli(dg)
    Bm=bernoulli(-dg)
    return -(Df/h)*(Bm*p[1:]-Bp*p[:-1])

def steady_dirichlet_sg(z,g,D,c_left=1.0,c_right=0.0):
    # Conservative Scharfetter-Gummel steady state in q = exp(g) p.
    # Resistance-network evaluation avoids p-space cancellation when the
    # stationary flux is very small.
    z,g,D,h=_validate(z,g,D)
    c_left=float(c_left); c_right=float(c_right)
    if not np.isfinite(c_left) or not np.isfinite(c_right) or c_left==c_right:
        raise ValueError("finite unequal boundary concentrations required")

    dg=np.diff(g)
    Df=np.sqrt(D[:-1]*D[1:])
    Bp=bernoulli(dg)
    if np.any(Bp<=0) or not np.all(np.isfinite(Bp)):
        raise FloatingPointError("invalid Bernoulli factors")

    logR=np.log(h)+g[:-1]-np.log(Df)-np.log(Bp)
    shift=float(np.max(logR))
    rscaled=np.exp(logR-shift)
    rsum=float(np.sum(rscaled))
    if not np.isfinite(rsum) or rsum<=0:
        raise FloatingPointError("invalid discrete resistance sum")

    q_left=float(np.exp(g[0])*c_left)
    q_right=float(np.exp(g[-1])*c_right)
    dq=q_left-q_right
    J=float(dq*np.exp(-shift)/rsum)
    if not np.isfinite(J):
        raise FloatingPointError("invalid stationary flux")

    frac=np.empty(len(z),dtype=np.float64)
    frac[0]=0.0
    frac[1:]=np.cumsum(rscaled)/rsum
    frac[-1]=1.0
    q=q_left-dq*frac
    p=q*np.exp(-g)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(p))):
        raise FloatingPointError("non-finite reconstructed solution")

    Jarr=np.full(len(z)-1,J,dtype=np.float64)
    P=J/(c_left-c_right)
    if not np.isfinite(P) or P<=0:
        raise FloatingPointError("non-positive/invalid permeability")

    return FPResult(
        p,Jarr,J,float(P),float(100.0*P),0.0,
        method="SCHAFETTER_GUMMEL_RESISTANCE_NETWORK"
    )

def equilibrium_zero_flux_diagnostic(z,g,D):
    z,g,D,_=_validate(z,g,D)
    shift=float(np.max(-g))
    p=np.exp(-g-shift)
    J=interface_flux_sg(z,g,D,p)
    dg=np.diff(g)
    Df=np.sqrt(D[:-1]*D[1:])
    h=float(z[1]-z[0])
    scale=float(np.max((Df/h)*np.maximum(p[:-1],p[1:])))
    rel=float(np.max(np.abs(J))/max(scale,np.finfo(np.float64).tiny))
    return {"max_abs_flux":float(np.max(np.abs(J))),"relative_zero_flux_residual":rel}
