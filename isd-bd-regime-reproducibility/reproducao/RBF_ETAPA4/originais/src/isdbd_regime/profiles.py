from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Geometry:
    H_nm: float; mu_inner_nm: float; mu_outer_nm: float; sigma_center_nm: float; sigma_inner_nm: float; sigma_outer_nm: float
    def validate(self)->None:
        if not self.H_nm>0: raise ValueError("H_nm must be positive")
        if not 0<self.mu_inner_nm<self.mu_outer_nm<self.H_nm: raise ValueError("Require 0 < mu_inner < mu_outer < H")
        if min(self.sigma_center_nm,self.sigma_inner_nm,self.sigma_outer_nm)<=0: raise ValueError("sigma must be positive")

@dataclass(frozen=True)
class ProfileCoefficients:
    center: float; inner: float; outer: float
    def as_array(self): return np.array([self.center,self.inner,self.outer],dtype=np.float64)

@dataclass(frozen=True)
class ProfileResult:
    z_nm: np.ndarray; g: np.ndarray; ell: np.ndarray; d_ratio: np.ndarray; D_nm2_ns: np.ndarray; g_prime_per_nm: np.ndarray; ell_prime_per_nm: np.ndarray; D_prime_nm_ns: np.ndarray; ito_drift_nm_ns: np.ndarray

def _z(x):
    z=np.asarray(x,dtype=np.float64)
    if z.ndim!=1 or len(z)<2 or not np.all(np.isfinite(z)): raise ValueError("invalid z grid")
    return z

def _qc(z,s): return np.exp(-(z*z)/(2*s*s))
def _dqc(z,s): return -(z/(s*s))*_qc(z,s)
def _qp(z,m,s): return np.exp(-((z-m)**2)/(2*s*s))+np.exp(-((z+m)**2)/(2*s*s))
def _dqp(z,m,s):
    a=np.exp(-((z-m)**2)/(2*s*s)); b=np.exp(-((z+m)**2)/(2*s*s))
    return -((z-m)/(s*s))*a-((z+m)/(s*s))*b

def anchored_center_basis(z_nm,H_nm,sigma_nm):
    z=_z(z_nm)
    if H_nm<=0 or sigma_nm<=0: raise ValueError("H and sigma positive")
    return _qc(z,sigma_nm)-_qc(np.array([H_nm]),sigma_nm)[0]
def anchored_center_basis_derivative(z_nm,sigma_nm):
    z=_z(z_nm)
    if sigma_nm<=0: raise ValueError("sigma positive")
    return _dqc(z,sigma_nm)
def anchored_pair_basis(z_nm,H_nm,mu_nm,sigma_nm):
    z=_z(z_nm)
    if H_nm<=0 or sigma_nm<=0 or not 0<mu_nm<H_nm: raise ValueError("invalid pair geometry")
    return _qp(z,mu_nm,sigma_nm)-_qp(np.array([H_nm]),mu_nm,sigma_nm)[0]
def anchored_pair_basis_derivative(z_nm,mu_nm,sigma_nm):
    z=_z(z_nm)
    if mu_nm<=0 or sigma_nm<=0: raise ValueError("mu and sigma positive")
    return _dqp(z,mu_nm,sigma_nm)

def _field(z,g,c):
    g.validate(); a=c.as_array()
    q=[anchored_center_basis(z,g.H_nm,g.sigma_center_nm),anchored_pair_basis(z,g.H_nm,g.mu_inner_nm,g.sigma_inner_nm),anchored_pair_basis(z,g.H_nm,g.mu_outer_nm,g.sigma_outer_nm)]
    dq=[anchored_center_basis_derivative(z,g.sigma_center_nm),anchored_pair_basis_derivative(z,g.mu_inner_nm,g.sigma_inner_nm),anchored_pair_basis_derivative(z,g.mu_outer_nm,g.sigma_outer_nm)]
    return a[0]*q[0]+a[1]*q[1]+a[2]*q[2], a[0]*dq[0]+a[1]*dq[1]+a[2]*dq[2]

def build_profile(z_nm,g_geometry,d_geometry,a,b,D_bulk_nm2_ns):
    z=_z(z_nm); g_geometry.validate(); d_geometry.validate()
    if D_bulk_nm2_ns<=0 or not np.isfinite(D_bulk_nm2_ns): raise ValueError("D_bulk positive finite")
    if g_geometry.H_nm!=d_geometry.H_nm: raise ValueError("H mismatch")
    H=g_geometry.H_nm
    if np.min(z)<-H or np.max(z)>H: raise ValueError("grid outside domain")
    gv,gp=_field(z,g_geometry,a); ell,ellp=_field(z,d_geometry,b)
    d=np.exp(ell); D=D_bulk_nm2_ns*d; Dp=D*ellp; drift=D*(ellp-gp)
    if not np.all(np.isfinite(gv)) or not np.all(np.isfinite(D)) or np.any(D<=0): raise FloatingPointError("invalid profile")
    return ProfileResult(z,gv,ell,d,D,gp,ellp,Dp,drift)

def evaluate_profile_acceptance(profile,g_min,g_max,d_min,d_max,atol=2e-12):
    if not g_min<g_max or not 0<d_min<=d_max or atol<0: raise ValueError("invalid bounds")
    glo=float(np.min(profile.g)); ghi=float(np.max(profile.g)); dlo=float(np.min(profile.d_ratio)); dhi=float(np.max(profile.d_ratio))
    gok=glo>=g_min-atol and ghi<=g_max+atol; dok=dlo>=d_min-atol and dhi<=d_max+atol
    return {"accepted":bool(gok and dok),"g_ok":bool(gok),"d_ok":bool(dok),"g_min_observed":glo,"g_max_observed":ghi,"d_min_observed":dlo,"d_max_observed":dhi}
