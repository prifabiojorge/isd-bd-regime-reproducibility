import numpy as np

from isdbd_regime.profiles import Geometry,ProfileCoefficients
from isdbd_regime.bd_flux import (
    first_exit_committor_counts,
    weighted_permeability_from_committor,
)

H=4.0
GG=Geometry(H,1.15,2.0,.62,.40,.29)
DG=Geometry(H,1.70,2.20,.40,.40,.40)
Z=ProfileCoefficients(0,0,0)

def test_flat_center_committor_smoke():
    c=first_exit_committor_counts(
        [0.0],4096,0.004,12345,"EM",GG,DG,Z,Z,.95,max_steps=50000
    )
    assert c.unresolved[0]==0
    assert abs(c.q_right[0]-0.5)<0.04

def test_weighted_estimator_synthetic():
    class C:
        start_z_nm=np.array([-2.0,0.0,2.0])
        total=np.array([10000,10000,10000])
        right=np.array([2500,5000,7500])
        unresolved=np.array([0,0,0])
        @property
        def q_right(self):
            return self.right/self.total
    # Flat D=1, H=4: I=(z+4), and P=1/8.
    I=np.array([2.0,4.0,6.0])
    P,se,q=weighted_permeability_from_committor(C(),I)
    assert abs(P-0.125)<5e-4
    assert se>0

def test_reproducible_fixed_seed():
    a=first_exit_committor_counts(
        [-1.0,1.0],512,0.004,777,"MILSTEIN",GG,DG,Z,Z,.95,max_steps=50000
    )
    b=first_exit_committor_counts(
        [-1.0,1.0],512,0.004,777,"MILSTEIN",GG,DG,Z,Z,.95,max_steps=50000
    )
    assert np.array_equal(a.right,b.right)
    assert np.array_equal(a.left,b.left)
    assert np.array_equal(a.unresolved,b.unresolved)
