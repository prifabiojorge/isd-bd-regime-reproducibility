from pathlib import Path
import json, os, sys, math
import numpy as np

root=Path(os.environ["ISDBD_ROOT"])
out=Path(os.environ["ISDBD_JOB_OUT"])
sys.path.insert(0,str(root/"src"))
sys.path.insert(0,str(Path(os.environ["ISDBD_STAGE"])/"src"))

from isdbd_regime.profiles import Geometry,ProfileCoefficients,build_profile,evaluate_profile_acceptance
from isdbd_regime.isd import isd_log_trapezoid
from isdbd_regime.fokker_planck import steady_dirichlet_sg,equilibrium_zero_flux_diagnostic

H=4.0
baseG=(1.15,2.0,.62,.40,.29)
baseD=(1.70,2.20,.40,.40,.40)

cases=[
 ("P1_MODERATE",(3,-1,.5),(-.3,-.4,-.2),baseG,baseD),
 ("P2_BASIN",(-4,2,1),(-.6,-.2,-.1),baseG,baseD),
 ("P3_BARRIER",(7,-3,2),(-.5,-.8,-.3),baseG,baseD),
 ("P4_NARROW_OUTER",(0,0,8),(0,0,-1.0),(1.15,2.0,.62,.40,.23),(1.70,2.20,.40,.40,.15)),
 ("P5_DEEP_CENTER",(-10,3,1),(-1.0,-.4,-.2),baseG,baseD),
]
grids=[801,1601,3201,6401]
rows=[]
flat=[]
equil=[]
transformed_diagnostic={}

for n in grids:
    z=np.linspace(-H,H,n)
    GG=Geometry(H,1.15,2.0,.62,.40,.29)
    DG=Geometry(H,1.70,2.20,.40,.40,.40)
    Z=ProfileCoefficients(0,0,0)
    for D0 in [.70,.95,1.20]:
        p=build_profile(z,GG,DG,Z,Z,D0)
        f=steady_dirichlet_sg(z,p.g,p.D_nm2_ns)
        expected=D0/8.0
        flat.append({"n":n,"D0":D0,"relative_error":abs(f.permeability_nm_per_ns-expected)/expected,
                     "flux_nonuniformity":f.relative_flux_nonuniformity})

for name,a,b,ggt,dgt in cases:
    GG=Geometry(H,*ggt)
    DG=Geometry(H,*dgt)
    for n in grids:
        z=np.linspace(-H,H,n)
        prof=build_profile(z,GG,DG,ProfileCoefficients(*a),ProfileCoefficients(*b),.95)
        if n==3201:
            v=evaluate_profile_acceptance(prof,-22.34161394,13.03260813,.09,1.0)
            if not v["accepted"]: raise RuntimeError(f"{name} failed physical profile filter")
            equil.append({"case":name,**equilibrium_zero_flux_diagnostic(z,prof.g,prof.D_nm2_ns)})
        fp=steady_dirichlet_sg(z,prof.g,prof.D_nm2_ns)
        isd=isd_log_trapezoid(z,prof.g,prof.D_nm2_ns)
        rel=abs(fp.permeability_nm_per_ns-isd.permeability_nm_per_ns)/isd.permeability_nm_per_ns
        rows.append({"case":name,"grid_points":n,"P_FP":fp.permeability_nm_per_ns,
                     "P_ISD":isd.permeability_nm_per_ns,"relative_difference":rel,
                     "flux_nonuniformity":fp.relative_flux_nonuniformity})

# Diagnostic for the corrected physical invariant:
# p can exceed the boundary concentration in a favorable free-energy well,
# while q=exp(g)p remains monotone.
zdiag=np.linspace(-H,H,3201)
GGdiag=Geometry(H,*baseG)
DGdiag=Geometry(H,*baseD)
profdiag=build_profile(
    zdiag,GGdiag,DGdiag,
    ProfileCoefficients(-4,2,1),
    ProfileCoefficients(-.6,-.2,-.1),
    .95
)
fpdiag=steady_dirichlet_sg(zdiag,profdiag.g,profdiag.D_nm2_ns)
qdiag=np.exp(profdiag.g)*fpdiag.p
transformed_diagnostic={
  "p_max":float(np.max(fpdiag.p)),
  "q_min":float(np.min(qdiag)),
  "q_max":float(np.max(qdiag)),
  "q_max_positive_increment":float(max(0.0,np.max(np.diff(qdiag)))),
  "flux_nonuniformity":float(fpdiag.relative_flux_nonuniformity),
}

agg={}
for n in grids:
    q=[x for x in rows if x["grid_points"]==n]
    agg[str(n)]={
      "max_relative_difference":max(x["relative_difference"] for x in q),
      "mean_relative_difference":sum(x["relative_difference"] for x in q)/len(q),
      "worst_case":max(q,key=lambda x:x["relative_difference"])["case"],
      "max_flux_nonuniformity":max(x["flux_nonuniformity"] for x in q),
    }

# empirical order of max cross-method discrepancy
orders=[]
for a,b in zip(grids[:-1],grids[1:]):
    ea=agg[str(a)]["max_relative_difference"]; eb=agg[str(b)]["max_relative_difference"]
    orders.append({"from":a,"to":b,"order_log2":math.log(ea/eb,2.0)})

metrics={
 "mission_id":"ISDBD-R2-B1-FOKKER-PLANCK-ENGINE-PILOT-001-FIX3",
 "solver":"SCHAFETTER_GUMMEL_FINITE_VOLUME",
 "pilot_cases":5,
 "grids":grids,
 "flat_controls":flat,
 "equilibrium_controls":equil,
 "transformed_variable_diagnostic":transformed_diagnostic,
 "fp_isd_rows":rows,
 "aggregate":agg,
 "orders":orders,
 "cross_method_tolerance_frozen":False,
 "FP_production_grid_frozen":False,
 "BD_computed":False,
 "MFPT_computed":False,
 "production_map_started":False,
}
(out/"r2b1_fp_pilot_metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")

print("__R2B1_RESULTS_BEGIN__")
print("pilot_cases=5")
print("grid_count=4")
print(f"flat_max_relerr={max(x['relative_error'] for x in flat):.12e}")
print(f"equilibrium_max_relative_zero_flux={max(x['relative_zero_flux_residual'] for x in equil):.12e}")
print(f"diagnostic_p_max={transformed_diagnostic['p_max']:.12e}")
print(f"diagnostic_q_min={transformed_diagnostic['q_min']:.12e}")
print(f"diagnostic_q_max={transformed_diagnostic['q_max']:.12e}")
print(f"diagnostic_q_max_positive_increment={transformed_diagnostic['q_max_positive_increment']:.12e}")
for n in grids:
    x=agg[str(n)]
    print(f"grid_{n}_max_fp_isd_rel={x['max_relative_difference']:.12e}")
    print(f"grid_{n}_mean_fp_isd_rel={x['mean_relative_difference']:.12e}")
    print(f"grid_{n}_worst_case={x['worst_case']}")
    print(f"grid_{n}_max_flux_nonuniformity={x['max_flux_nonuniformity']:.12e}")
for x in orders:
    print(f"order_{x['from']}_{x['to']}={x['order_log2']:.9f}")
print("cross_method_tolerance_frozen=NO")
print("FP_production_grid_frozen=NO")
print("BD_computed=NO")
print("MFPT_computed=NO")
print("production_map_started=NO")
print("__R2B1_RESULTS_END__")
