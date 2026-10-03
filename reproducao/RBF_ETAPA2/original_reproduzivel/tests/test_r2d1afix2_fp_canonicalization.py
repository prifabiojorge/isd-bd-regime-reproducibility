import numpy as np
from scipy.stats import qmc
from isdbd_regime.profiles import Geometry,ProfileCoefficients,build_profile
from isdbd_regime.isd import isd_log_trapezoid
from isdbd_regime.fokker_planck import steady_dirichlet_sg

H=4.0
ALO=-16.756210454869; AHI=9.774456098674
BLO=-1.805959206489; BHI=0.0
def L(u,a,b): return a+(b-a)*u

def test_extreme_profile_3936_regression():
    u=qmc.Sobol(d=17,scramble=False).random_base2(13)[3936]
    a=ProfileCoefficients(*(L(u[j],ALO,AHI) for j in range(3)))
    gg=Geometry(H,L(u[3],.80,1.50),L(u[4],1.75,2.25),L(u[5],.55,.70),L(u[6],.23,.70),L(u[7],.23,.35))
    b=ProfileCoefficients(*(L(u[j],BLO,BHI) for j in range(8,11)))
    dg=Geometry(H,L(u[11],1.50,1.90),L(u[12],2.00,2.40),L(u[13],.15,.80),L(u[14],.15,.80),L(u[15],.15,.80))
    db=L(u[16],.70,1.20)
    z=np.linspace(-H,H,6401)
    p=build_profile(z,gg,dg,a,b,db)
    fp=steady_dirichlet_sg(z,p.g,p.D_nm2_ns)
    isd=isd_log_trapezoid(z,p.g,p.D_nm2_ns)
    rel=abs(fp.permeability_nm_per_ns-isd.permeability_nm_per_ns)/isd.permeability_nm_per_ns
    assert fp.method=="SCHAFETTER_GUMMEL_RESISTANCE_NETWORK"
    assert rel<5e-5
