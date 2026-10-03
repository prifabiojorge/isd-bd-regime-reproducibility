import numpy as np

from isdbd_regime.profiles import Geometry,ProfileCoefficients,build_profile
from isdbd_regime.brownian import (
    analytic_fields,reflect_interval,em_step,milstein_step
)

H=4.0
GG=Geometry(H,1.15,2.0,.62,.40,.29)
DG=Geometry(H,1.70,2.20,.40,.40,.40)
A=ProfileCoefficients(3.0,-1.0,.5)
B=ProfileCoefficients(-.3,-.4,-.2)

def test_analytic_fields_match_profile_engine():
    z=np.linspace(-H,H,1001)
    p=build_profile(z,GG,DG,A,B,.95)
    q=analytic_fields(z,GG,DG,A,B,.95)
    assert np.max(np.abs(p.g-q.g))<2e-14
    assert np.max(np.abs(p.ell-q.ell))<2e-14
    assert np.max(np.abs(p.D_nm2_ns-q.D_nm2_ns))<2e-14
    assert np.max(np.abs(p.g_prime_per_nm-q.g_prime_per_nm))<5e-14
    assert np.max(np.abs(p.ell_prime_per_nm-q.ell_prime_per_nm))<5e-14
    assert np.max(np.abs(p.D_prime_nm_ns-q.D_prime_nm_ns))<5e-14
    assert np.max(np.abs(p.ito_drift_nm_ns-q.ito_drift_nm_ns))<8e-14

def test_reflection_always_inside():
    x=np.array([-123.4,-12.1,-4.0,-3.2,0.0,3.9,4.0,9.2,101.7])
    y=reflect_interval(x,H)
    assert np.all(y>=-H)
    assert np.all(y<=H)

def test_reflection_simple_overshoot():
    x=np.array([4.2,-4.2,4.7,-4.7])
    y=reflect_interval(x,H)
    expected=np.array([3.8,-3.8,3.3,-3.3])
    assert np.max(np.abs(y-expected))<2e-15

def test_flat_em_and_milstein_same_with_same_rng():
    Z=ProfileCoefficients(0,0,0)
    z=np.linspace(-3.5,3.5,1000)
    r1=np.random.default_rng(123)
    r2=np.random.default_rng(123)
    a=em_step(z,1e-3,r1,GG,DG,Z,Z,.95)
    b=milstein_step(z,1e-3,r2,GG,DG,Z,Z,.95)
    assert np.max(np.abs(a-b))<2e-15

def test_steps_remain_finite_and_inside():
    z=np.linspace(-H,H,2000)
    for method in ("EM","MILSTEIN"):
        rng=np.random.default_rng(44)
        step=em_step if method=="EM" else milstein_step
        y=step(z,2e-3,rng,GG,DG,A,B,.95)
        assert np.all(np.isfinite(y))
        assert np.all(y>=-H)
        assert np.all(y<=H)
