import numpy as np
from isdbd_regime.profiles import Geometry, ProfileCoefficients, build_profile
from isdbd_regime.isd import isd_log_trapezoid
from isdbd_regime.fokker_planck import (
    bernoulli,
    steady_dirichlet_sg,
    equilibrium_zero_flux_diagnostic,
)

H=4.0; L=8.0
GG=Geometry(H,1.15,2.0,0.62,0.40,0.29)
DG=Geometry(H,1.70,2.20,0.40,0.40,0.40)
Z=ProfileCoefficients(0,0,0)

def grid(n=3201): return np.linspace(-H,H,n)

def test_bernoulli_identity():
    x=np.array([-2.,-.2,0.,.2,2.])
    b=bernoulli(x)
    bm=bernoulli(-x)
    assert np.max(np.abs(bm-np.exp(x)*b))<5e-15

def test_flat_fp_exact():
    z=grid()
    for D0 in [0.70,0.95,1.20]:
        p=build_profile(z,GG,DG,Z,Z,D0)
        r=steady_dirichlet_sg(z,p.g,p.D_nm2_ns,1.0,0.0)
        expected=D0/L
        assert abs(r.permeability_nm_per_ns-expected)/expected<2e-11
        expected_p=np.linspace(1.0,0.0,len(z))
        assert np.max(np.abs(r.p-expected_p))<2e-11

def test_equilibrium_is_preserved():
    z=grid()
    p=build_profile(z,GG,DG,ProfileCoefficients(7,-3,2),ProfileCoefficients(-.5,-.8,-.3),.95)
    q=equilibrium_zero_flux_diagnostic(z,p.g,p.D_nm2_ns)
    assert q["relative_zero_flux_residual"]<2e-14

def test_nonflat_fp_isd_smoke():
    z=grid()
    p=build_profile(z,GG,DG,ProfileCoefficients(3,-1,.5),ProfileCoefficients(-.3,-.4,-.2),.95)
    fp=steady_dirichlet_sg(z,p.g,p.D_nm2_ns)
    isd=isd_log_trapezoid(z,p.g,p.D_nm2_ns)
    rel=abs(fp.permeability_nm_per_ns-isd.permeability_nm_per_ns)/isd.permeability_nm_per_ns
    # Implementation smoke guard only, NOT a frozen FP/ISD scientific tolerance.
    assert rel<1e-3

def test_positive_solution_and_transformed_monotonicity():
    z=grid()
    prof=build_profile(
        z,GG,DG,
        ProfileCoefficients(-4,2,1),
        ProfileCoefficients(-.6,-.2,-.1),
        .95
    )
    fp=steady_dirichlet_sg(z,prof.g,prof.D_nm2_ns)

    # For J = -D(p' + g' p), q = exp(g) p satisfies
    # q' = -J exp(g)/D. For positive left-to-right flux,
    # q must decrease monotonically. p itself may exceed its
    # boundary value inside a favorable free-energy well (g < 0).
    q=np.exp(prof.g)*fp.p

    assert np.min(fp.p)>=-1e-12
    assert abs(q[0]-1.0)<2e-12
    assert abs(q[-1]-0.0)<2e-12
    assert np.min(q)>=-2e-12
    assert np.max(q)<=1.0+2e-12
    assert np.max(np.diff(q))<=2e-12
    assert fp.relative_flux_nonuniformity<5e-9
