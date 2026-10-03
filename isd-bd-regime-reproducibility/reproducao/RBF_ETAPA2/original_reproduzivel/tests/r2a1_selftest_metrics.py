import json,os
from pathlib import Path
import numpy as np
from isdbd_regime import *
stage=Path(os.environ["ISDBD_STAGE"]); H=4.0; L=8.0; z=np.linspace(-H,H,3201)
G=Geometry(H,1.15,2.0,0.62,0.40,0.29); D=Geometry(H,1.70,2.20,0.40,0.40,0.40); Z=ProfileCoefficients(0,0,0)
flat=[]
for D0 in [.70,.95,1.20]:
 p=build_profile(z,G,D,Z,Z,D0); a=isd_log_trapezoid(z,p.g,p.D_nm2_ns); b=isd_direct(z,p.g,p.D_nm2_ns); e=D0/L
 flat.append({"D0":D0,"expected":e,"stable":a.permeability_nm_per_ns,"direct":b.permeability_nm_per_ns,"stable_relerr":abs(a.permeability_nm_per_ns-e)/e,"direct_relerr":abs(b.permeability_nm_per_ns-e)/e})
cases=[("M1",ProfileCoefficients(3,-1,.5),ProfileCoefficients(-.3,-.4,-.2),.70),("M2",ProfileCoefficients(-4,2,1),ProfileCoefficients(-.6,-.2,-.1),.95),("M3",ProfileCoefficients(7,-3,2),ProfileCoefficients(-.5,-.8,-.3),1.20)]
moderate=[]
for name,a,b,D0 in cases:
 p=build_profile(z,G,D,a,b,D0); x=isd_log_trapezoid(z,p.g,p.D_nm2_ns); y=isd_direct(z,p.g,p.D_nm2_ns)
 moderate.append({"case":name,"relative_difference":abs(x.permeability_nm_per_ns-y.permeability_nm_per_ns)/y.permeability_nm_per_ns})
p=build_profile(z,G,D,ProfileCoefficients(3,-2,1),ProfileCoefficients(-.5,-.7,-.2),.95)
m={"grid_points":3201,"flat_controls":flat,"moderate_comparison":moderate,"structural":{"g_boundary":max(abs(float(p.g[0])),abs(float(p.g[-1]))),"ell_boundary":max(abs(float(p.ell[0])),abs(float(p.ell[-1]))),"even_g":float(np.max(np.abs(p.g-p.g[::-1]))),"even_ell":float(np.max(np.abs(p.ell-p.ell[::-1]))),"odd_gp":float(np.max(np.abs(p.g_prime_per_nm+p.g_prime_per_nm[::-1]))),"odd_ellp":float(np.max(np.abs(p.ell_prime_per_nm+p.ell_prime_per_nm[::-1]))),"Dprime_identity":float(np.max(np.abs(p.D_prime_nm_ns-p.D_nm2_ns*p.ell_prime_per_nm))),"min_D":float(np.min(p.D_nm2_ns))},"MFPT_computed":False,"FP_computed":False,"BD_computed":False,"production_map_started":False,"production_tolerance_frozen":False}
(stage/"data/model/r2a1_selftest_metrics.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("__R2A1_METRICS_BEGIN__")
print("grid_points=3201")
print("flat_control_count=3")
print(f"flat_max_stable_relative_error={max(x['stable_relerr'] for x in flat):.12e}")
print(f"flat_max_direct_relative_error={max(x['direct_relerr'] for x in flat):.12e}")
print("moderate_comparison_count=3")
print(f"moderate_max_log_vs_direct_relative_difference={max(x['relative_difference'] for x in moderate):.12e}")
for k,v in m['structural'].items(): print(f"{k}={v:.12e}")
print("MFPT_computed=NO"); print("FP_computed=NO"); print("BD_computed=NO"); print("production_map_started=NO"); print("production_tolerance_frozen=NO")
print("__R2A1_METRICS_END__")
