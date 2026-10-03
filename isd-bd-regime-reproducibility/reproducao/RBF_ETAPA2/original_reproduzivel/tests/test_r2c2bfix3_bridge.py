import numpy as np

from isdbd_regime.profiles import Geometry,ProfileCoefficients
from isdbd_regime.bd_flux import (
    first_exit_committor_counts_bridge,
    boundary_linear_permeability_from_committor,
)

H=4.0
GG=Geometry(H,1.15,2.0,.62,.40,.29)
DG=Geometry(H,1.70,2.20,.40,.40,.40)
Z=ProfileCoefficients(0,0,0)

def test_bridge_flat_committor_smoke():
    starts=-H+np.array([.25,.35,.45,.55,.65])
    c,diag=first_exit_committor_counts_bridge(
        starts,8192,.004,7123,"EM",GG,DG,Z,Z,.95,max_steps=100000
    )
    assert np.sum(c.unresolved)==0
    exact=(starts+H)/(2*H)
    assert np.max(np.abs(c.q_right-exact))<0.025
    assert diag["bridge_total_hits"]>0

def test_bridge_flat_boundary_slope_smoke():
    starts=-H+np.array([.25,.35,.45,.55,.65])
    c,diag=first_exit_committor_counts_bridge(
        starts,8192,.004,8134,"EM",GG,DG,Z,Z,.95,max_steps=100000
    )
    r=boundary_linear_permeability_from_committor(c,.95,-H,.65)
    exact=.95/(2*H)
    assert abs(r["P_nm_ns"]-exact)/exact<0.05
    assert r["points_used"]==5
