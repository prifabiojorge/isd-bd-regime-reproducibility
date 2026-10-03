from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import logsumexp

@dataclass(frozen=True)
class ISDResult:
    resistance_ns_per_nm: float; permeability_nm_per_ns: float; permeability_cm_per_s: float; log_resistance: float; log_permeability_native: float; method: str

def _inputs(z_nm,g,D):
    z=np.asarray(z_nm,dtype=np.float64); gg=np.asarray(g,dtype=np.float64); dd=np.asarray(D,dtype=np.float64)
    if z.ndim!=1 or len(z)<2 or gg.shape!=z.shape or dd.shape!=z.shape: raise ValueError("shape mismatch")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(gg)) or not np.all(np.isfinite(dd)) or np.any(dd<=0): raise ValueError("invalid inputs")
    dz=np.diff(z)
    if np.any(dz<=0): raise ValueError("z not increasing")
    dz0=float(dz[0])
    if not np.allclose(dz,dz0,rtol=2e-12,atol=5e-15): raise ValueError("uniform grid required")
    return z,gg,dd,dz0

def _pack(logR,method):
    logP=-float(logR); R=float(math.exp(logR)); P=float(math.exp(logP))
    return ISDResult(R,P,100.0*P,float(logR),logP,method)

def isd_log_trapezoid(z_nm,g,D_nm2_ns):
    z,gg,D,dz=_inputs(z_nm,g,D_nm2_ns)
    logw=np.zeros_like(z); logw[0]=math.log(0.5); logw[-1]=math.log(0.5)
    logR=math.log(dz)+float(logsumexp(logw+gg-np.log(D)))
    return _pack(logR,"LOG_TRAPEZOID")

def isd_direct(z_nm,g,D_nm2_ns):
    z,gg,D,_=_inputs(z_nm,g,D_nm2_ns)
    integ=np.exp(gg)/D
    if not np.all(np.isfinite(integ)): raise FloatingPointError("direct integrand nonfinite")
    R=float(np.trapezoid(integ,z))
    if not math.isfinite(R) or R<=0: raise FloatingPointError("invalid resistance")
    return _pack(math.log(R),"DIRECT_TRAPEZOID")
