from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .brownian import analytic_fields
from .profiles import Geometry, ProfileCoefficients

@dataclass(frozen=True)
class ExitCounts:
    start_z_nm: np.ndarray
    total: np.ndarray
    right: np.ndarray
    left: np.ndarray
    unresolved: np.ndarray
    steps_executed: int

    @property
    def q_right(self):
        return self.right / self.total

def _raw_step(z,dt,rng,method,g_geometry,d_geometry,a,b,D_bulk):
    f=analytic_fields(z,g_geometry,d_geometry,a,b,D_bulk)
    xi=rng.standard_normal(len(z))
    trial=z+f.ito_drift_nm_ns*dt+np.sqrt(2.0*f.D_nm2_ns*dt)*xi
    if method=="MILSTEIN":
        trial=trial+0.5*f.D_prime_nm_ns*dt*(xi*xi-1.0)
    return trial

def first_exit_committor_counts(
    start_z_nm,
    paths_per_start,
    dt_ns,
    seed,
    method,
    g_geometry,
    d_geometry,
    a,
    b,
    D_bulk_nm2_ns,
    max_steps=250000,
):
    starts=np.asarray(start_z_nm,dtype=np.float64)
    if starts.ndim!=1 or len(starts)<1 or not np.all(np.isfinite(starts)):
        raise ValueError("start_z_nm must be finite 1D")
    H=float(g_geometry.H_nm)
    if np.any(starts<=-H) or np.any(starts>=H):
        raise ValueError("starts must be strictly inside the domain")
    n=int(paths_per_start)
    if n<1 or n!=paths_per_start:
        raise ValueError("paths_per_start must be a positive integer")
    if not np.isfinite(dt_ns) or dt_ns<=0:
        raise ValueError("dt must be positive finite")
    method=str(method).upper()
    if method not in {"EM","MILSTEIN"}:
        raise ValueError("method must be EM or MILSTEIN")

    groups=len(starts)
    z=np.repeat(starts,n).astype(np.float64)
    gid=np.repeat(np.arange(groups,dtype=np.int32),n)
    active=np.ones(len(z),dtype=bool)
    right=np.zeros(groups,dtype=np.int64)
    left=np.zeros(groups,dtype=np.int64)
    rng=np.random.default_rng(int(seed))

    steps=0
    for steps in range(1,int(max_steps)+1):
        idx=np.flatnonzero(active)
        if len(idx)==0:
            break
        trial=_raw_step(
            z[idx],dt_ns,rng,method,
            g_geometry,d_geometry,a,b,D_bulk_nm2_ns
        )
        hitL=trial<=-H
        hitR=trial>=H
        if np.any(hitL):
            np.add.at(left,gid[idx[hitL]],1)
        if np.any(hitR):
            np.add.at(right,gid[idx[hitR]],1)
        done=hitL|hitR
        if np.any(done):
            active[idx[done]]=False
        stay=~done
        if np.any(stay):
            z[idx[stay]]=trial[stay]

    unresolved=np.bincount(gid[active],minlength=groups).astype(np.int64)
    total=np.full(groups,n,dtype=np.int64)
    return ExitCounts(starts,total,right,left,unresolved,steps)

def weighted_permeability_from_committor(
    counts,
    resistance_left_ns_per_nm,
):
    I=np.asarray(resistance_left_ns_per_nm,dtype=np.float64)
    if I.shape!=counts.start_z_nm.shape or np.any(I<=0) or not np.all(np.isfinite(I)):
        raise ValueError("invalid resistance_left")
    if np.any(counts.unresolved!=0):
        raise RuntimeError("unresolved first-exit trajectories present")

    q=counts.q_right.astype(np.float64)
    n=counts.total.astype(np.float64)

    # Jeffreys-stabilized binomial variance to avoid zero-weight pathologies.
    qj=(counts.right+0.5)/(n+1.0)
    var=qj*(1.0-qj)/(n+1.0)
    w=1.0/var

    denom=float(np.sum(w*I*I))
    if denom<=0 or not np.isfinite(denom):
        raise FloatingPointError("invalid weighted regression denominator")
    P=float(np.sum(w*I*q)/denom)
    se=float(np.sqrt(1.0/denom))
    return P,se,q

def first_exit_committor_counts_bridge(
    start_z_nm,
    paths_per_start,
    dt_ns,
    seed,
    method,
    g_geometry,
    d_geometry,
    a,
    b,
    D_bulk_nm2_ns,
    max_steps=250000,
):
    # First-exit committor with Brownian-bridge correction for crossings
    # that occur between two discrete endpoints which both remain inside.
    #
    # Conditional on endpoints for locally constant D, the bridge crossing
    # probability for a boundary is exp[-distance_start*distance_end/(D*dt)].
    # This is exact for constant D and is used here first on the FLAT control.
    starts=np.asarray(start_z_nm,dtype=np.float64)
    if starts.ndim!=1 or len(starts)<1 or not np.all(np.isfinite(starts)):
        raise ValueError("start_z_nm must be finite 1D")
    H=float(g_geometry.H_nm)
    if np.any(starts<=-H) or np.any(starts>=H):
        raise ValueError("starts must be strictly inside the domain")
    n=int(paths_per_start)
    if n<1 or n!=paths_per_start:
        raise ValueError("paths_per_start must be a positive integer")
    if not np.isfinite(dt_ns) or dt_ns<=0:
        raise ValueError("dt must be positive finite")
    method=str(method).upper()
    if method not in {"EM","MILSTEIN"}:
        raise ValueError("method must be EM or MILSTEIN")

    groups=len(starts)
    z=np.repeat(starts,n).astype(np.float64)
    gid=np.repeat(np.arange(groups,dtype=np.int32),n)
    active=np.ones(len(z),dtype=bool)
    right=np.zeros(groups,dtype=np.int64)
    left=np.zeros(groups,dtype=np.int64)
    rng=np.random.default_rng(int(seed))

    steps=0
    bridge_left_hits=0
    bridge_right_hits=0

    for steps in range(1,int(max_steps)+1):
        idx=np.flatnonzero(active)
        if len(idx)==0:
            break

        zz=z[idx]
        f=analytic_fields(zz,g_geometry,d_geometry,a,b,D_bulk_nm2_ns)
        xi=rng.standard_normal(len(idx))
        trial=zz+f.ito_drift_nm_ns*dt_ns+np.sqrt(2.0*f.D_nm2_ns*dt_ns)*xi
        if method=="MILSTEIN":
            trial=trial+0.5*f.D_prime_nm_ns*dt_ns*(xi*xi-1.0)

        hitL=trial<=-H
        hitR=trial>=H

        inside=~(hitL|hitR)
        if np.any(inside):
            ii=np.flatnonzero(inside)
            z0=zz[ii]
            z1=trial[ii]
            Dloc=f.D_nm2_ns[ii]

            expoL=-((z0+H)*(z1+H))/(Dloc*dt_ns)
            expoR=-((H-z0)*(H-z1))/(Dloc*dt_ns)
            pL=np.exp(np.minimum(0.0,expoL))
            pR=np.exp(np.minimum(0.0,expoR))

            # For the present 8-nm domain these events are mutually exclusive
            # to machine precision at audited dt. Still guard against a sum
            # slightly above one by proportional renormalization.
            ps=pL+pR
            over=ps>1.0
            if np.any(over):
                pL[over]/=ps[over]
                pR[over]/=ps[over]

            u=rng.random(len(ii))
            bL=u<pL
            bR=(~bL) & (u < (pL+pR))

            if np.any(bL):
                hitL[ii[bL]]=True
                bridge_left_hits+=int(np.sum(bL))
            if np.any(bR):
                hitR[ii[bR]]=True
                bridge_right_hits+=int(np.sum(bR))

        if np.any(hitL):
            np.add.at(left,gid[idx[hitL]],1)
        if np.any(hitR):
            np.add.at(right,gid[idx[hitR]],1)

        done=hitL|hitR
        if np.any(done):
            active[idx[done]]=False
        stay=~done
        if np.any(stay):
            z[idx[stay]]=trial[stay]

    unresolved=np.bincount(gid[active],minlength=groups).astype(np.int64)
    total=np.full(groups,n,dtype=np.int64)
    out=ExitCounts(starts,total,right,left,unresolved,steps)
    return out,{
      "bridge_left_hits":bridge_left_hits,
      "bridge_right_hits":bridge_right_hits,
      "bridge_total_hits":bridge_left_hits+bridge_right_hits,
    }

def boundary_linear_permeability_from_committor(
    counts,
    D_bulk_nm2_ns,
    left_boundary_nm,
    max_offset_nm,
):
    # Weighted linear boundary slope through exact q(-H)=0.
    if np.any(counts.unresolved != 0):
        raise RuntimeError("unresolved first-exit trajectories present")
    x=np.asarray(counts.start_z_nm-float(left_boundary_nm),dtype=np.float64)
    max_offset=float(max_offset_nm)
    edge_tol=64.0*np.finfo(np.float64).eps*max(1.0,abs(max_offset))
    use=(x>0) & (x<=max_offset+edge_tol)
    if np.sum(use)<3:
        raise ValueError("at least three boundary points are required")

    x=x[use]
    q=counts.q_right.astype(np.float64)[use]
    n=counts.total.astype(np.float64)[use]
    r=counts.right.astype(np.float64)[use]
    qj=(r+0.5)/(n+1.0)
    var=qj*(1.0-qj)/(n+1.0)
    w=1.0/var
    denom=float(np.sum(w*x*x))
    slope=float(np.sum(w*x*q)/denom)
    slope_se=float(np.sqrt(1.0/denom))
    residual=q-slope*x
    chi2=float(np.sum(w*residual*residual))
    dof=int(len(x)-1)
    return {
      "P_nm_ns":float(D_bulk_nm2_ns*slope),
      "P_SE_nm_ns":float(D_bulk_nm2_ns*slope_se),
      "points_used":int(len(x)),
      "reduced_chi2":float(chi2/dof),
    }
