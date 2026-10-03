from pathlib import Path
import csv,gzip,json,math,os,shutil,sys,time
import numpy as np
import yaml
from scipy.stats import qmc

ROOT=Path(os.environ["ISDBD_ROOT"])
OUT=Path(os.environ["ISDBD_JOB"])
sys.path.insert(0,str(ROOT/"src"))

from isdbd_regime.profiles import Geometry,ProfileCoefficients,build_profile,evaluate_profile_acceptance
from isdbd_regime.isd import isd_log_trapezoid
from isdbd_regime.fokker_planck import steady_dirichlet_sg

t0=time.time()

design=yaml.safe_load((ROOT/"config/r2d2a_production_ensemble_design.yaml").read_text(encoding="utf-8"))
assert design["status"]=="PRODUCTION_ENSEMBLE_DESIGN_FROZEN"
assert int(design["candidate_design"]["primary_candidate_N"])==65536
assert int(design["numerical_validation"]["ISD_refinement"]["subset_size"])==1024
assert int(design["numerical_validation"]["FP_crosscheck"]["subset_size"])==128

with (ROOT/"data/literature/literature_bounds.csv").open(newline="",encoding="utf-8") as f:
    lit=list(csv.DictReader(f))
B={r["bound_id"]:r for r in lit}
lo=lambda k: float(B[k]["frozen_low"])
hi=lambda k: float(B[k]["frozen_high"])

policy=yaml.safe_load((ROOT/"config/r1d2b_production_coefficient_policy.yaml").read_text(encoding="utf-8"))
A_LO,A_HI=[float(x) for x in policy["coefficient_bounds"]["a_j"]]
B_LO,B_HI=[float(x) for x in policy["coefficient_bounds"]["b_j"]]

H=lo("R1D-H"); L=2*H
G_LO,G_HI=lo("R1D-g-PROFILE"),hi("R1D-g-PROFILE")
D_MIN,D_MAX=lo("R1D-DMIN-RATIO"),hi("R1D-DMIN-RATIO")
ETA_HI=hi("R1D-ETA")
DB_LO,DB_HI=lo("R1D-DBULK"),hi("R1D-DBULK")
MU_GI=(lo("R1D-MU-G-INNER"),hi("R1D-MU-G-INNER"))
MU_GO=(lo("R1D-MU-G-OUTER"),hi("R1D-MU-G-OUTER"))
SG_C=(lo("R1D-SIGMA-G-CENTER"),hi("R1D-SIGMA-G-CENTER"))
SG_I=(lo("R1D-SIGMA-G-INNER"),hi("R1D-SIGMA-G-INNER"))
SG_O=(lo("R1D-SIGMA-G-OUTER"),hi("R1D-SIGMA-G-OUTER"))
MU_DI=(lo("R1D-MU-D-INNER"),hi("R1D-MU-D-INNER"))
MU_DO=(lo("R1D-MU-D-OUTER"),hi("R1D-MU-D-OUTER"))
SD=(lo("R1D-SIGMA-D"),hi("R1D-SIGMA-D"))

def lin(u,a,b): return a+(b-a)*u

def decode(u):
    a=ProfileCoefficients(*(lin(u[j],A_LO,A_HI) for j in range(3)))
    gg=Geometry(H,lin(u[3],*MU_GI),lin(u[4],*MU_GO),
                lin(u[5],*SG_C),lin(u[6],*SG_I),lin(u[7],*SG_O))
    b=ProfileCoefficients(*(lin(u[j],B_LO,B_HI) for j in range(8,11)))
    dg=Geometry(H,lin(u[11],*MU_DI),lin(u[12],*MU_DO),
                lin(u[13],*SD),lin(u[14],*SD),lin(u[15],*SD))
    db=lin(u[16],DB_LO,DB_HI)
    return a,gg,b,dg,db

def evaluate_record(sobol_index,a,gg,b,dg,db,N):
    z=np.linspace(-H,H,N,dtype=np.float64)
    zeros=np.zeros_like(z)
    p=build_profile(z,gg,dg,a,b,db)
    Dflat=np.full_like(z,db)

    r0=isd_log_trapezoid(z,zeros,Dflat)
    rg=isd_log_trapezoid(z,p.g,Dflat)
    rd=isd_log_trapezoid(z,zeros,p.D_nm2_ns)
    rgd=isd_log_trapezoid(z,p.g,p.D_nm2_ns)

    logR0=float(r0.log_resistance)
    logRG=float(rg.log_resistance)
    logRD=float(rd.log_resistance)
    logRGD=float(rgd.log_resistance)
    AG=logRG-logR0
    AD=logRD-logR0
    interaction=logRGD-logRG-logRD+logR0
    CG=AG+0.5*interaction
    CD=AD+0.5*interaction
    deltaC=AG-AD
    maxg=float(np.max(p.g))
    ming=float(np.min(p.g))
    mind=float(np.min(p.d_ratio))
    maxd=float(np.max(p.d_ratio))
    eta=-math.log10(mind)
    wG=math.exp(AG-maxg)
    wD=math.exp(AD-math.log(10.0)*eta)
    rho=math.log(wG/wD)
    peak_abs_z=float(abs(z[int(np.argmax(p.g-p.ell))]))

    return {
      "sobol_index":int(sobol_index),
      "a_center":float(a.center),"a_inner":float(a.inner),"a_outer":float(a.outer),
      "mu_g_inner_nm":float(gg.mu_inner_nm),"mu_g_outer_nm":float(gg.mu_outer_nm),
      "sigma_g_center_nm":float(gg.sigma_center_nm),"sigma_g_inner_nm":float(gg.sigma_inner_nm),"sigma_g_outer_nm":float(gg.sigma_outer_nm),
      "b_center":float(b.center),"b_inner":float(b.inner),"b_outer":float(b.outer),
      "mu_d_inner_nm":float(dg.mu_inner_nm),"mu_d_outer_nm":float(dg.mu_outer_nm),
      "sigma_d_center_nm":float(dg.sigma_center_nm),"sigma_d_inner_nm":float(dg.sigma_inner_nm),"sigma_d_outer_nm":float(dg.sigma_outer_nm),
      "D_bulk_nm2_ns":float(db),
      "g_min":ming,"g_max":maxg,"d_min":mind,"d_max":maxd,"eta":eta,
      "P_nm_ns":float(rgd.permeability_nm_per_ns),
      "P_cm_s":float(rgd.permeability_cm_per_s),
      "log10_P_cm_s":float(math.log10(rgd.permeability_cm_per_s)),
      "logR0":logR0,"logRG":logRG,"logRD":logRD,"logRGD":logRGD,
      "AG":AG,"AD":AD,"interaction":interaction,
      "CG":CG,"CD":CD,"deltaC":deltaC,
      "wG":wG,"wD":wD,"rho":rho,
      "resistance_peak_abs_z_nm":peak_abs_z,
      "evaluation_grid_N":int(N),
      "numerical_refinement_applied":bool(N==12801),
    }

# Generate accepted production rows at default N=3201.
U=qmc.Sobol(d=17,scramble=False).random_base2(16)
zfilter=np.linspace(-H,H,3201,dtype=np.float64)

rows=[]
rejected=0
decoded={}

for sobol_i,u in enumerate(U):
    a,gg,b,dg,db=decode(u)
    p=build_profile(zfilter,gg,dg,a,b,db)
    if not evaluate_profile_acceptance(p,G_LO,G_HI,D_MIN,D_MAX)["accepted"]:
        rejected+=1
        continue
    decoded[int(sobol_i)]=(a,gg,b,dg,db)
    rows.append(evaluate_record(sobol_i,a,gg,b,dg,db,3201))

accepted=len(rows)
if accepted!=56926 or rejected!=8610:
    raise RuntimeError(f"production reproduction mismatch accepted={accepted} rejected={rejected}")

# Frozen deterministic 1024-subset constructor.
AGarr=np.array([r["AG"] for r in rows],dtype=float)
ADarr=np.array([r["AD"] for r in rows],dtype=float)
AGmin,AGmax=float(np.min(AGarr)),float(np.max(AGarr))
ADmin,ADmax=float(np.min(ADarr)),float(np.max(ADarr))

def add_unique(seq,seen,indices,target):
    for idx in indices:
        i=int(idx)
        if i not in seen:
            seen.add(i); seq.append(i)
            if len(seq)>=target: return True
    return False

def rank_spaced(key,k):
    order=np.argsort(np.array([r[key] for r in rows]),kind="mergesort")
    pos=np.unique(np.linspace(0,accepted-1,min(k,accepted),dtype=int))
    return order[pos]

def near_boundary(k):
    order=np.argsort(np.array([abs(r["deltaC"]) for r in rows]),kind="mergesort")
    return order[:min(k,accepted)]

def plane_representatives(nx,ny):
    xs=AGarr; ys=ADarr
    xed=np.linspace(AGmin,AGmax,nx+1); yed=np.linspace(ADmin,ADmax,ny+1)
    reps=[]
    for ix in range(nx):
        for iy in range(ny):
            mask=np.where(
                (xs>=xed[ix]) & ((xs<xed[ix+1]) if ix<nx-1 else (xs<=xed[ix+1])) &
                (ys>=yed[iy]) & ((ys<yed[iy+1]) if iy<ny-1 else (ys<=yed[iy+1]))
            )[0]
            if len(mask)==0: continue
            cx=.5*(xed[ix]+xed[ix+1]); cy=.5*(yed[iy]+yed[iy+1])
            sx=max(xed[ix+1]-xed[ix],1e-15); sy=max(yed[iy+1]-yed[iy],1e-15)
            d=((xs[mask]-cx)/sx)**2+((ys[mask]-cy)/sy)**2
            reps.append(int(mask[int(np.argmin(d))]))
    return reps

def build_subset(target,rank_each,near_k,plane_n):
    seq=[]; seen=set()
    for key in ["AG","AD","deltaC","rho","log10_P_cm_s"]:
        if add_unique(seq,seen,rank_spaced(key,rank_each),target): return seq
    if add_unique(seq,seen,plane_representatives(plane_n,plane_n),target): return seq
    if add_unique(seq,seen,near_boundary(near_k),target): return seq
    add_unique(seq,seen,np.arange(accepted),target)
    if len(seq)!=target: raise RuntimeError(f"subset mismatch {len(seq)}")
    return seq

ref_idx=build_subset(1024,150,128,10)

# Apply frozen numerical escalation.
ref_audit=[]
flagged=[]
for idx in ref_idx:
    r=rows[idx]
    a,gg,b,dg,db=decoded[r["sobol_index"]]
    r64=evaluate_record(r["sobol_index"],a,gg,b,dg,db,6401)
    e32_64=abs(r["P_nm_ns"]-r64["P_nm_ns"])/r64["P_nm_ns"]
    rec={
      "sobol_index":r["sobol_index"],
      "rel_3201_6401":float(e32_64),
      "rel_6401_12801":"",
      "P3201":r["P_nm_ns"],
      "P6401":r64["P_nm_ns"],
      "P12801":"",
      "escalated":False,
      "escalation_pass":"",
    }
    if e32_64>1e-7:
        r128=evaluate_record(r["sobol_index"],a,gg,b,dg,db,12801)
        e64_128=abs(r64["P_nm_ns"]-r128["P_nm_ns"])/r128["P_nm_ns"]
        rec["rel_6401_12801"]=float(e64_128)
        rec["P12801"]=r128["P_nm_ns"]
        rec["escalated"]=True
        rec["escalation_pass"]=bool(e64_128<=1e-7)
        flagged.append((idx,r["sobol_index"],e32_64,e64_128,r128))
    ref_audit.append(rec)

primary_fail_count=len(flagged)
escalation_fail_count=sum(1 for x in flagged if x[3]>1e-7)
if primary_fail_count!=1:
    raise RuntimeError(f"expected exactly one primary refinement failure, got {primary_fail_count}")
if flagged[0][1]!=36312:
    raise RuntimeError(f"expected Sobol 36312, got {flagged[0][1]}")
if escalation_fail_count!=0:
    raise RuntimeError("frozen 6401->12801 escalation guard failed")

# Replace flagged production records with fully recomputed N=12801 records.
for idx,sobol_i,e32_64,e64_128,r128 in flagged:
    if not evaluate_profile_acceptance(
        build_profile(
            np.linspace(-H,H,12801),
            decoded[sobol_i][1],decoded[sobol_i][3],
            decoded[sobol_i][0],decoded[sobol_i][2],decoded[sobol_i][4]
        ),
        G_LO,G_HI,D_MIN,D_MAX
    )["accepted"]:
        raise RuntimeError(f"refined profile {sobol_i} failed full-profile filter at N=12801")
    rows[idx]=r128

refined_profiles=[r["sobol_index"] for r in rows if r["numerical_refinement_applied"]]
if refined_profiles!=[36312]:
    raise RuntimeError(f"unexpected refined set {refined_profiles}")

# Recompute final counts and architecture invariants after refinement.
thermo=sum(r["deltaC"]>0 for r in rows)
diffusive=sum(r["deltaC"]<0 for r in rows)
near=sum(abs(r["deltaC"])<=0.10 for r in rows)
max_delta_identity=max(abs(r["deltaC"]-(r["AG"]-r["AD"])) for r in rows)
max_proxy_shape_identity=max(abs(r["deltaC"]-(r["g_max"]-math.log(10.0)*r["eta"]+r["rho"])) for r in rows)
finite_rho=all(math.isfinite(r["rho"]) for r in rows)

AGarr=np.array([r["AG"] for r in rows],dtype=float)
ADarr=np.array([r["AD"] for r in rows],dtype=float)
xarr=np.array([r["g_max"] for r in rows],dtype=float)
etaarr=np.array([r["eta"] for r in rows],dtype=float)

AGedges=np.linspace(float(np.min(AGarr)),float(np.max(AGarr)),33)
ADedges=np.linspace(float(np.min(ADarr)),float(np.max(ADarr)),33)
hexact,_,_=np.histogram2d(AGarr,ADarr,bins=[AGedges,ADedges])
exact_occupied=int(np.sum(hexact>0))
exact_fraction=exact_occupied/1024.0

xedges=np.linspace(0.0,G_HI,33)
yedges=np.linspace(0.0,ETA_HI,33)
hproxy,_,_=np.histogram2d(xarr,etaarr,bins=[xedges,yedges])
proxy_occupied=int(np.sum(hproxy>0))

coverage_pass=(
    accepted>=50000 and thermo>=5000 and diffusive>=5000 and near>=500
    and exact_fraction>=0.70 and proxy_occupied>=500 and finite_rho
)

# Rebuild deterministic FP subset on final production rows.
AGmin,AGmax=float(np.min(AGarr)),float(np.max(AGarr))
ADmin,ADmax=float(np.min(ADarr)),float(np.max(ADarr))
fp_idx=build_subset(128,12,32,6)

fp_rows=[]
solver_methods=set()
z64=np.linspace(-H,H,6401,dtype=np.float64)
for idx in fp_idx:
    r=rows[idx]
    a,gg,b,dg,db=decoded[r["sobol_index"]]
    p=build_profile(z64,gg,dg,a,b,db)
    fp=steady_dirichlet_sg(z64,p.g,p.D_nm2_ns)
    solver_methods.add(fp.method)
    isd64=isd_log_trapezoid(z64,p.g,p.D_nm2_ns)
    rel_prod=abs(fp.permeability_nm_per_ns-r["P_nm_ns"])/r["P_nm_ns"]
    rel_same=abs(fp.permeability_nm_per_ns-isd64.permeability_nm_per_ns)/isd64.permeability_nm_per_ns
    fp_rows.append({
      "sobol_index":r["sobol_index"],
      "production_grid_N":r["evaluation_grid_N"],
      "deltaC":r["deltaC"],
      "FP6401_nm_ns":float(fp.permeability_nm_per_ns),
      "production_P_nm_ns":r["P_nm_ns"],
      "ISD6401_nm_ns":float(isd64.permeability_nm_per_ns),
      "relative_FP6401_vs_production":float(rel_prod),
      "relative_FP6401_vs_ISD6401":float(rel_same),
      "method":fp.method,
    })

max_fp=max(r["relative_FP6401_vs_production"] for r in fp_rows)
max_fp_same=max(r["relative_FP6401_vs_ISD6401"] for r in fp_rows)
canonical_solver=(solver_methods=={"SCHAFETTER_GUMMEL_RESISTANCE_NETWORK"})

max_primary=max(r["rel_3201_6401"] for r in ref_audit)
max_escalation=max(float(r["rel_6401_12801"]) for r in ref_audit if r["escalated"])

numerics_pass=(
    primary_fail_count==1
    and escalation_fail_count==0
    and max_escalation<=1e-7
    and max_fp<=5e-5
    and canonical_solver
)

production_pass=(
    coverage_pass and numerics_pass
    and max_delta_identity<1e-12
    and max_proxy_shape_identity<1e-12
)

# Final deterministic compressed dataset.
fields=list(rows[0].keys())
plain=OUT/"r2d2bfix2_production_ensemble.csv"
with plain.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
    w.writeheader(); w.writerows(rows)

gzpath=OUT/"r2d2bfix2_production_ensemble.csv.gz"
with plain.open("rb") as fi, gzpath.open("wb") as fo:
    with gzip.GzipFile(filename="",mode="wb",fileobj=fo,mtime=0) as gz:
        shutil.copyfileobj(fi,gz,1024*1024)
plain.unlink()

with (OUT/"r2d2bfix2_isd_refinement_audit.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(ref_audit[0].keys()),lineterminator="\n")
    w.writeheader(); w.writerows(ref_audit)

with (OUT/"r2d2bfix2_fp_crosscheck_subset.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(fp_rows[0].keys()),lineterminator="\n")
    w.writeheader(); w.writerows(fp_rows)

metrics={
 "mission_id":"ISDBD-R2-D2B-MATERIALIZE-PRODUCTION-ENSEMBLE-WITH-REFINEMENT-001-FIX2",
 "sampler":{"family":"Sobol","scramble":False,"dimension":17,"candidate_N":65536},
 "counts":{
   "candidate":65536,"accepted":accepted,"rejected":rejected,
   "thermodynamic":thermo,"diffusive":diffusive,
   "near_boundary_abs_deltaC_le_0p10":near,
 },
 "numerical_refinement":{
   "subset_size":1024,
   "primary_grid_N":3201,
   "refinement_grid_N":6401,
   "escalation_grid_N":12801,
   "primary_guard":1e-7,
   "primary_fail_count":primary_fail_count,
   "primary_fail_sobol_ids":[x[1] for x in flagged],
   "max_rel_3201_6401":max_primary,
   "escalation_fail_count":escalation_fail_count,
   "max_rel_6401_12801_for_escalated":max_escalation,
   "refined_production_sobol_ids":refined_profiles,
   "refinement_policy_applied":"flagged_profiles_use_full_N12801_record",
 },
 "coverage":{
   "exact_control_32x32_occupied_bins":exact_occupied,
   "exact_control_32x32_occupied_fraction":exact_fraction,
   "proxy_32x32_occupied_bins":proxy_occupied,
   "coverage_pass":bool(coverage_pass),
 },
 "FP_crosscheck":{
   "subset_size":len(fp_rows),
   "solver_methods":sorted(solver_methods),
   "max_relative_FP6401_vs_production":max_fp,
   "max_relative_FP6401_vs_ISD6401":max_fp_same,
   "guard":5e-5,
   "canonical_solver":bool(canonical_solver),
 },
 "architecture_invariants":{
   "max_abs_deltaC_minus_AG_minus_AD":max_delta_identity,
   "max_abs_deltaC_minus_proxy_shape":max_proxy_shape_identity,
   "finite_rho_all":bool(finite_rho),
 },
 "numerics_pass":bool(numerics_pass),
 "production_pass":bool(production_pass),
 "production_dataset_frozen":bool(production_pass),
 "production_map_computed":False,
 "neutral_band_frozen":False,
 "MFPT_computed":False,
 "novelty_claim_made":False,
 "runtime_seconds":time.time()-t0,
}
(OUT/"r2d2bfix2_production_ensemble_metrics.json").write_text(
    json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8"
)

print("__RESULT_BEGIN__")
print("candidate_N=65536")
print(f"accepted_N={accepted}")
print(f"rejected_N={rejected}")
print(f"thermodynamic_count={thermo}")
print(f"diffusive_count={diffusive}")
print(f"near_boundary_abs_deltaC_le_0p10={near}")
print(f"primary_fail_count={primary_fail_count}")
print("primary_fail_sobol_ids="+",".join(str(x[1]) for x in flagged))
print(f"max_rel_3201_6401={max_primary:.12e}")
print(f"escalation_fail_count={escalation_fail_count}")
print(f"max_rel_6401_12801={max_escalation:.12e}")
print("refined_production_sobol_ids="+",".join(str(x) for x in refined_profiles))
print(f"exact_control_occupied_bins_32x32={exact_occupied}")
print(f"exact_control_occupied_fraction={exact_fraction:.12e}")
print(f"proxy_occupied_bins_32x32={proxy_occupied}")
print(f"FP_crosscheck_subset={len(fp_rows)}")
print("FP_solver_methods="+",".join(sorted(solver_methods)))
print(f"max_rel_FP6401_vs_production={max_fp:.12e}")
print(f"max_rel_FP6401_vs_ISD6401={max_fp_same:.12e}")
print(f"max_identity_deltaC_AG_AD={max_delta_identity:.12e}")
print(f"max_identity_proxy_shape={max_proxy_shape_identity:.12e}")
print(f"coverage_pass={'YES' if coverage_pass else 'NO'}")
print(f"numerics_pass={'YES' if numerics_pass else 'NO'}")
print(f"production_pass={'YES' if production_pass else 'NO'}")
print(f"production_dataset_frozen={'YES' if production_pass else 'NO'}")
print("production_map_computed=NO")
print("MFPT_computed=NO")
print("novelty_claim_made=NO")
print("__RESULT_END__")

if not production_pass:
    raise SystemExit(7)
