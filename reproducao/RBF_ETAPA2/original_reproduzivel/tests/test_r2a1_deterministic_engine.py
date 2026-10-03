import numpy as np
import pytest
from isdbd_regime import *
H=4.0; L=8.0
G=Geometry(H,1.15,2.0,0.62,0.40,0.29)
D=Geometry(H,1.70,2.20,0.40,0.40,0.40)
Z=ProfileCoefficients(0.0,0.0,0.0)
def grid(): return np.linspace(-H,H,3201,dtype=np.float64)

def test_anchor_and_parity():
    z=np.array([-H,H],dtype=np.float64)
    assert np.array_equal(anchored_center_basis(z,H,0.62),np.zeros(2))
    assert np.array_equal(anchored_pair_basis(z,H,1.15,0.40),np.zeros(2))
    z=grid()
    q=anchored_pair_basis(z,H,1.15,0.40); dq=anchored_pair_basis_derivative(z,1.15,0.40)
    assert np.max(np.abs(q-q[::-1]))<8e-15
    assert np.max(np.abs(dq+dq[::-1]))<8e-14

def test_profile_identities():
    z=grid(); p=build_profile(z,G,D,ProfileCoefficients(3,-2,1),ProfileCoefficients(-.5,-.7,-.2),.95)
    assert np.max(np.abs(p.g-p.g[::-1]))<2e-14
    assert np.max(np.abs(p.ell-p.ell[::-1]))<2e-14
    assert np.max(np.abs(p.g_prime_per_nm+p.g_prime_per_nm[::-1]))<2e-13
    assert np.max(np.abs(p.ell_prime_per_nm+p.ell_prime_per_nm[::-1]))<2e-13
    assert np.max(np.abs(p.D_prime_nm_ns-p.D_nm2_ns*p.ell_prime_per_nm))<2e-15
    assert np.all(p.D_nm2_ns>0)

@pytest.mark.parametrize("D0",[0.70,0.95,1.20])
def test_flat_analytic(D0):
    z=grid(); p=build_profile(z,G,D,Z,Z,D0); expected=D0/L
    a=isd_log_trapezoid(z,p.g,p.D_nm2_ns); b=isd_direct(z,p.g,p.D_nm2_ns)
    assert abs(a.permeability_nm_per_ns-expected)/expected<3e-13
    assert abs(b.permeability_nm_per_ns-expected)/expected<3e-13
    assert abs(a.permeability_cm_per_s-100*expected)/(100*expected)<3e-13

def test_filter_and_comparator():
    z=grid(); good=build_profile(z,G,D,ProfileCoefficients(2,-1,.5),ProfileCoefficients(-.2,-.3,-.1),.95)
    assert evaluate_profile_acceptance(good,-22.34161394,13.03260813,.09,1.0)["accepted"]
    bad=build_profile(z,G,D,ProfileCoefficients(40,40,40),ProfileCoefficients(1,1,1),.95)
    assert not evaluate_profile_acceptance(bad,-22.34161394,13.03260813,.09,1.0)["accepted"]
    for a,b,D0 in [(ProfileCoefficients(3,-1,.5),ProfileCoefficients(-.3,-.4,-.2),.70),(ProfileCoefficients(-4,2,1),ProfileCoefficients(-.6,-.2,-.1),.95),(ProfileCoefficients(7,-3,2),ProfileCoefficients(-.5,-.8,-.3),1.20)]:
        p=build_profile(z,G,D,a,b,D0); x=isd_log_trapezoid(z,p.g,p.D_nm2_ns); y=isd_direct(z,p.g,p.D_nm2_ns)
        assert abs(x.permeability_nm_per_ns-y.permeability_nm_per_ns)/y.permeability_nm_per_ns<3e-13

def test_guardrails():
    z=grid()
    with pytest.raises(ValueError): Geometry(4,2,1,.5,.5,.5).validate()
    with pytest.raises(ValueError): build_profile(z,G,D,Z,Z,-1)
    with pytest.raises(ValueError): isd_log_trapezoid(z,np.zeros_like(z),np.zeros_like(z))
