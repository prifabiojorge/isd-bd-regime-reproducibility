"""Mesma família gaussiana/ISD, em lotes; validada contra os módulos originais."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import qmc
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"originais/src"))
from isdbd_regime.profiles import Geometry, ProfileCoefficients, build_profile, evaluate_profile_acceptance
from isdbd_regime.isd import isd_log_trapezoid

PROTOCOL = json.loads((ROOT/"PROTOCOLO_ETAPA4.json").read_text())
BOUNDS = pd.read_csv(ROOT/"originais/PARAMETROS_17.csv",float_precision="round_trip")
FIELDS = BOUNDS.csv_field.to_list()
LO, HI = BOUNDS.low.to_numpy(float), BOUNDS.high.to_numpy(float)
POLICY = yaml.safe_load((ROOT/"originais/config/r1d2b_production_coefficient_policy.yaml").read_text())
LIMITS = {k:POLICY["mandatory_full_profile_acceptance"][k] for k in ["g_min","g_max","d_min","d_max"]}
H, L, NGRID, TAU = 4., 8., 3201, 2e-12
FINAL = ["g_min","g_max","d_min","d_max","eta","AG","AD","interaction","CG","CD",
         "deltaC","rho","wG","wD","P_cm_s","log10_P_cm_s","proxy_margin",
         "resistance_peak_abs_z_nm","D_bulk_nm2_ns"]
FILTER = ["filter_g_min","filter_g_max","filter_d_min","filter_d_max"]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def verify_inputs():
    manifest=json.loads((ROOT/"evidencias/entradas_sha256.json").read_text())
    for rel, sha in manifest.items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha, rel


def scenario_bounds(spec):
    lo, hi = LO.copy(), HI.copy()
    ix = spec.get("fields", [])
    change = spec.get("change")
    if change == "scale_bounds":
        lo[ix] *= spec["factor"]; hi[ix] *= spec["factor"]
    elif change == "central_half":
        span = hi[ix]-lo[ix]
        lo[ix] += .25*span; hi[ix] -= .25*span
    elif change == "lower_half":
        hi[ix] = .5*(lo[ix]+hi[ix])
    elif change == "upper_half":
        lo[ix] = .5*(lo[ix]+hi[ix])
    assert np.all(lo>=LO) and np.all(hi<=HI) and np.all(lo<hi)
    return lo,hi


def points(spec):
    m=spec.get("m",16)
    scramble=spec.get("scramble",False)
    kwargs=dict(d=17,scramble=scramble,bits=30,optimization=None)
    if scramble: kwargs["rng"]=int(spec["seed"])
    engine=qmc.Sobol(**kwargs)
    all_u=engine.random_base2(m)
    start=spec.get("start_index",0)
    u=all_u[start:]
    ids=np.arange(start,len(all_u),dtype=np.uint32)
    lo,hi=scenario_bounds(spec)
    p=lo+(hi-lo)*u
    assert np.all(p>=lo) and np.all(p<hi)
    assert np.all(p[:,3]>0) and np.all(p[:,3]<p[:,4]) and np.all(p[:,4]<H)
    assert np.all(p[:,11]<p[:,12]) and np.all(p[:,12]<H)
    assert np.all(p[:,[5,6,7,13,14,15]]>0) and np.all(p[:,16]>0)
    meta=dict(spec,dimension=17,bits=30,optimization=None,discard_first=False,thin=False,
              input_points_sha256=hashlib.sha256(u.tobytes()).hexdigest(),
              input_parameter_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
              lower_bounds=lo.tolist(),upper_bounds=hi.tolist(),fields=FIELDS,
              rng_bit_generator=type(engine.rng.bit_generator).__name__ if scramble else None,
              namespace="(case_id, point_index); IDs de novos perfis não são registros originais")
    return ids,u,p,meta


def source_record(p,n=3201):
    z=np.linspace(-H,H,n,dtype=np.float64)
    prof=build_profile(z,Geometry(H,*p[3:8]),Geometry(H,*p[11:16]),
                       ProfileCoefficients(*p[:3]),ProfileCoefficients(*p[8:11]),float(p[16]))
    flat=np.full_like(z,p[16]);zero=np.zeros_like(z)
    r0=isd_log_trapezoid(z,zero,flat)
    rg=isd_log_trapezoid(z,prof.g,flat)
    rd=isd_log_trapezoid(z,zero,prof.D_nm2_ns)
    rgd=isd_log_trapezoid(z,prof.g,prof.D_nm2_ns)
    ag=rg.log_resistance-r0.log_resistance
    ad=rd.log_resistance-r0.log_resistance
    interaction=rgd.log_resistance-rg.log_resistance-rd.log_resistance+r0.log_resistance
    gmin,gmax=float(prof.g.min()),float(prof.g.max())
    dmin,dmax=float(prof.d_ratio.min()),float(prof.d_ratio.max())
    eta=-math.log10(dmin);dc=ag-ad;m=gmax-math.log(10)*eta
    wg,wd=math.exp(ag-gmax),math.exp(ad-math.log(10)*eta)
    return dict(g_min=gmin,g_max=gmax,d_min=dmin,d_max=dmax,eta=eta,
                AG=ag,AD=ad,interaction=interaction,CG=ag+interaction/2,CD=ad+interaction/2,
                deltaC=dc,rho=math.log(wg/wd),wG=wg,wD=wd,P_cm_s=rgd.permeability_cm_per_s,
                log10_P_cm_s=math.log10(rgd.permeability_cm_per_s),proxy_margin=m,
                resistance_peak_abs_z_nm=abs(float(z[np.argmax(prof.g-prof.ell)])),
                D_bulk_nm2_ns=float(p[16]),
                source_accepted=evaluate_profile_acceptance(prof,**LIMITS)["accepted"])


def evaluate(ids,p,batch=64):
    n=len(p)
    out={k:np.full(n,np.nan) for k in FINAL+FILTER}
    out.update(point_index=ids.copy(), fail_bits=np.zeros(n,dtype=np.uint8),
               accepted=np.zeros(n,dtype=bool),evaluation_grid_N=np.full(n,NGRID,dtype=np.uint16))
    z=np.linspace(-H,H,NGRID)[None,:]
    lw=np.zeros(NGRID);lw[[0,-1]]=math.log(.5)
    log_normalizer=math.log(NGRID-1)
    for start in range(0,n,batch):
        end=min(start+batch,n);a=p[start:end];v=lambda k:a[:,k,None]
        def field(c,ci,co,mi,mo,sc,si,so):
            def central(s):
                return np.exp(-(z*z)/(2*s*s))-np.exp(-(H*H)/(2*s*s))
            def pair(mu,s):
                return (np.exp(-((z-mu)**2)/(2*s*s))+np.exp(-((z+mu)**2)/(2*s*s))
                        -(np.exp(-((H-mu)**2)/(2*s*s))+np.exp(-((H+mu)**2)/(2*s*s))))
            return c*central(sc)+ci*pair(mi,si)+co*pair(mo,so)
        g=field(v(0),v(1),v(2),v(3),v(4),v(5),v(6),v(7))
        ell=field(v(8),v(9),v(10),v(11),v(12),v(13),v(14),v(15))
        d=np.exp(ell)
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(d)) and np.all(d>0)
        gmin,gmax=g.min(axis=1),g.max(axis=1)
        dmin,dmax=d.min(axis=1),d.max(axis=1)
        flags=np.zeros(len(a),dtype=np.uint8)
        for bit,ok in [(1,gmin>=LIMITS["g_min"]-TAU),(2,gmax<=LIMITS["g_max"]+TAU),
                       (4,dmin>=LIMITS["d_min"]-TAU),(8,dmax<=LIMITS["d_max"]+TAU)]:
            flags[~ok]|=bit
        accepted=flags==0
        ag=logsumexp(g+lw,axis=1)-log_normalizer
        ad=logsumexp(-ell+lw,axis=1)-log_normalizer
        total=logsumexp(g-ell+lw,axis=1)-log_normalizer
        interaction=total-ag-ad
        dc=ag-ad
        eta=-np.log10(dmin)
        wg=np.exp(ag-gmax);wd=np.exp(ad-math.log(10)*eta)
        values=dict(g_min=gmin,g_max=gmax,d_min=dmin,d_max=dmax,eta=eta,
                    AG=ag,AD=ad,interaction=interaction,CG=ag+interaction/2,CD=ad+interaction/2,
                    deltaC=dc,rho=np.log(wg/wd),wG=wg,wD=wd,
                    P_cm_s=100*a[:,16]/L*np.exp(-total),
                    log10_P_cm_s=(math.log(100/L)+np.log(a[:,16])-total)/math.log(10),
                    proxy_margin=gmax-math.log(10)*eta,
                    resistance_peak_abs_z_nm=np.abs(z[0,np.argmax(g-ell,axis=1)]),
                    D_bulk_nm2_ns=a[:,16])
        for k,val in values.items():
            out[k][start:end]=np.where(accepted,val,np.nan)
        for k,val in zip(FILTER,[gmin,gmax,dmin,dmax]):out[k][start:end]=val
        out["fail_bits"][start:end]=flags;out["accepted"][start:end]=accepted
    return out


def apply_source(out,i,r,n):
    for k in FINAL:out[k][i]=r[k]
    out["evaluation_grid_N"][i]=n


def guard_labels(out,p):
    mask=out["accepted"]&((np.abs(out["deltaC"])<=1e-5)|(np.abs(out["proxy_margin"])<=1e-5))
    logs=[]
    for i in np.flatnonzero(mask):
        old=[float(out["deltaC"][i]),float(out["proxy_margin"][i])]
        r64=source_record(p[i],6401);r128=source_record(p[i],12801)
        assert r64["source_accepted"] and r128["source_accepted"]
        for key in ["deltaC","proxy_margin"]:
            assert np.sign(r64[key])==np.sign(r128[key]), f"Rótulo ainda não resolvido: {key}, índice {out['point_index'][i]}"
        apply_source(out,i,r128,12801)
        logs.append(dict(point_index=int(out["point_index"][i]),before_delta=old[0],before_proxy=old[1],
                         refined_delta=r128["deltaC"],refined_proxy=r128["proxy_margin"],
                         abs_delta_difference_6401_12801=abs(r64["deltaC"]-r128["deltaC"]),
                         abs_proxy_difference_6401_12801=abs(r64["proxy_margin"]-r128["proxy_margin"]),
                         integrated_label_changed=int(np.sign(old[0]))!=int(np.sign(r128["deltaC"])),
                         proxy_label_changed=int(np.sign(old[1]))!=int(np.sign(r128["proxy_margin"]))))
    return logs


def original_spot_checks(out,p):
    ids=np.arange(len(p))
    selected=list(np.unique(np.linspace(0,len(p)-1,min(96,len(p)),dtype=int)))
    ok=np.flatnonzero(out["accepted"])
    for key in ["deltaC","proxy_margin"]:
        selected+=ok[np.argsort(np.abs(out[key][ok]),kind="mergesort")[:16]].tolist()
    selected=list(dict.fromkeys(selected))
    maximum={k:0. for k in FINAL if k!="resistance_peak_abs_z_nm"}
    for i in selected:
        r=source_record(p[i],int(out["evaluation_grid_N"][i]))
        assert bool(out["accepted"][i])==r["source_accepted"]
        if out["accepted"][i]:
            for k in maximum:
                err=abs(float(out[k][i])-r[k])
                if k=="P_cm_s":err/=r[k]
                maximum[k]=max(maximum[k],err)
        else:
            for k,src in zip(FILTER,["g_min","g_max","d_min","d_max"]):
                assert abs(float(out[k][i])-r[src])<=2e-12
    assert max(maximum.values())<=2e-12, maximum
    return dict(N=len(selected),point_indices=[int(out["point_index"][i]) for i in selected],
                maximum_errors=maximum,P_error_relative=True,passed=True)


def classify(out):
    valid=out["accepted"]
    out["integrated_class"]=np.where(valid,np.sign(np.nan_to_num(out["deltaC"])),9).astype(np.int8)
    out["proxy_class"]=np.where(valid,np.sign(np.nan_to_num(out["proxy_margin"])),9).astype(np.int8)
    if np.any(valid):
        res=np.abs(out["deltaC"][valid]-out["proxy_margin"][valid]-out["rho"][valid])
        assert float(res.max())<=2e-12
        assert np.all(out["AD"][valid]>=-2e-12)
        assert np.all(out["wG"][valid]<=1+2e-12) and np.all(out["wD"][valid]<=1+2e-12)


def load_case(name):
    with np.load(ROOT/"dados"/(name+".npz"),allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}


def run_case(spec):
    start=time.monotonic();case_id=spec["id"]
    print("Início "+case_id,flush=True)
    ids,u,p,meta=points(spec)
    out=evaluate(ids,p)
    guards=guard_labels(out,p)
    checks=original_spot_checks(out,p)
    classify(out)
    np.savez_compressed(ROOT/"dados"/(case_id+".npz"),**out)
    meta.update(candidate_N=len(ids),accepted_N=int(out["accepted"].sum()),elapsed_seconds=time.monotonic()-start,
                source_spot_checks=checks,guard_refinements=guards,filter_grid_N=3201,filter_atol=TAU,
                scalar_results_dtype="float64",model_form_changed=False,
                parameter_domain_changed=spec.get("kind")=="domain_remap",original_table_changed=False)
    write_json(ROOT/"evidencias"/(case_id+".json"),meta)
    print(f"Concluído {case_id}: aceitos={meta['accepted_N']}; segundos={meta['elapsed_seconds']:.1f}; guardas={len(guards)}",flush=True)
    return case_id
