"""Quarteto contrafactual BD: contagens auditáveis, Monte Carlo e controles de dt."""
import ctypes
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import platform
import subprocess
import time
import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta
from numerica import *
from isdbd_regime.brownian import analytic_fields
from isdbd_regime.bd_flux import ExitCounts,boundary_linear_permeability_from_committor


def native():
    library=ctypes.CDLL(str(ROOT/"codigo/brownian_kernel.so"))
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
    ints=np.ctypeslib.ndpointer(dtype=np.int64,flags="C_CONTIGUOUS")
    library.fields_many.argtypes=[arr,arr,ctypes.c_int,arr]
    library.raw_step.argtypes=[arr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int]
    library.raw_step.restype=ctypes.c_double
    library.simulate.argtypes=[arr,arr,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_uint64,
                              ctypes.c_int,ctypes.c_int,ints,ints,arr]
    library.simulate.restype=ctypes.c_int
    return library


def prepare():
    cfg=PROTOCOL["bd"]
    subprocess.run(["g++","-O3","-std=c++17","-fPIC","-shared",str(ROOT/"codigo/brownian_kernel.cpp"),
                    "-o",str(ROOT/"codigo/brownian_kernel.so")],check=True)
    version=subprocess.run(["g++","--version"],capture_output=True,text=True,check=True).stdout
    write_json(ROOT/"evidencias/compilacao_bd.json",dict(compiler=version,flags=["-O3","-std=c++17","-fPIC","-shared"],
               fast_math=False,python=platform.python_version(),architecture=platform.machine(),
               kernel_cpp_sha256=hashlib.sha256((ROOT/"codigo/brownian_kernel.cpp").read_bytes()).hexdigest(),
               kernel_so_sha256=hashlib.sha256((ROOT/"codigo/brownian_kernel.so").read_bytes()).hexdigest(),
               random_generator=cfg["generator"]))
    # As geometrias e amplitudes-base vêm dos testes recuperados, não de uma inferência dos resumos BD.
    a=np.asarray(cfg["baseline_a"],float);b=np.asarray(cfg["baseline_b"],float)
    theta=np.array([*a,*cfg["baseline_geometry"][:5],*b,*cfg["baseline_geometry"][5:],cfg["D_bulk_nm2_ns"]])
    assert len(theta)==17
    def delta(scale):
        current=theta.copy();current[:3]=scale*a
        return adaptive_quad(current,eps=1e-12)["deltaC"]
    scale=brentq(delta,*cfg["g_scale_root_interval"],xtol=cfg["root_absolute_tolerance"],rtol=1e-14)
    calibrated=theta.copy();calibrated[:3]=scale*a
    controls=[]
    for name in cfg["states"]:
        p=calibrated.copy()
        if name in ["FLAT","D_ONLY"]:p[:3]=0.
        if name in ["FLAT","G_ONLY"]:p[8:11]=0.
        reference=adaptive_quad(p,eps=1e-12)
        fp=fp_direct(p,12801)
        controls.append(dict(case_id=name,theta=p.tolist(),ISD=reference,FP=fp))
    lib=native();checks=[]
    z=np.linspace(-H,H,2001,dtype=np.float64)
    for control in controls:
        p=np.array(control["theta"],float)
        gg,dg,a0,b0,bulk=geometries(p)
        py=analytic_fields(z,gg,dg,a0,b0,bulk)
        out=np.empty((len(z),7),float);lib.fields_many(p,z,len(z),out)
        expected=np.array([py.g,py.ell,py.D_nm2_ns,py.g_prime_per_nm,py.ell_prime_per_nm,
                           py.D_prime_nm_ns,py.ito_drift_nm_ns]).T
        error=float(np.max(np.abs(out-expected)))
        assert error<1e-11,(control["case_id"],error)
        step_error=0.
        for i in [0,150,333,555,1000,1555,1999]:
            for xi in [-2.1,-.25,0.,.75,2.3]:
                for method in [0,1]:
                    dt=.002
                    expected_step=z[i]+py.ito_drift_nm_ns[i]*dt+math.sqrt(2*py.D_nm2_ns[i]*dt)*xi
                    if method:expected_step+=.5*py.D_prime_nm_ns[i]*dt*(xi*xi-1.)
                    step_error=max(step_error,abs(lib.raw_step(p,float(z[i]),dt,xi,method)-expected_step))
        assert step_error<1e-12
        boundary=np.linspace(-H,-H+max(cfg["offsets_nm"]),1001)
        bp=analytic_fields(boundary,gg,dg,a0,b0,bulk)
        maxg=float(np.max(np.abs(bp.g)));maxD=float(np.max(np.abs(bp.D_nm2_ns/bulk-1.)))
        assert maxg<=cfg["maximum_abs_g_boundary"] and maxD<=cfg["maximum_relative_D_deviation_boundary"]
        control["boundary_window"]={"max_abs_g":maxg,"max_relative_D_deviation":maxD}
        checks.append(dict(case_id=control["case_id"],field_max_absolute_error=error,raw_step_max_absolute_error=step_error))
    write_json(ROOT/"evidencias/validacao_kernel_bd.json",dict(status="PASS",nodes_per_control=len(z),rows=checks,
               functions_originals_unchanged=True))
    write_json(ROOT/"dados/controle_fronteira_bd.json",dict(g_scale=scale,root_delta=delta(scale),
               numerical_root_not_analytic_exact_tie=True,added_to_original_production=False,controls=controls))
    return controls


def run(spec):
    t=time.monotonic();cfg=PROTOCOL["bd"]
    control=spec["control"];p=np.array(control["theta"],float)
    starts=np.array(cfg["offsets_nm"],float)-H
    counts=np.empty((len(starts),4),np.int64);diagnostics=np.empty(5,np.int64);maximum=np.empty(1,float)
    horizon=cfg["initial_horizon_ns"];lib=native()
    while True:
        code=lib.simulate(p,starts,len(starts),cfg["paths_per_start"],spec["dt_ns"],spec["seed"],
                          int(spec["method"]=="MILSTEIN"),round(horizon/spec["dt_ns"]),counts,diagnostics,maximum)
        assert code==0,code
        unresolved=int(counts[:,3].sum())
        if not unresolved or horizon>=cfg["escalation_horizon_ns"]:break
        horizon=cfg["escalation_horizon_ns"]
    assert np.all(counts[:,0]==counts[:,1:].sum(axis=1))
    result=dict(case_id=control["case_id"],method=spec["method"],dt_ns=spec["dt_ns"],seed=spec["seed"],
                paths_per_start=cfg["paths_per_start"],starts_nm=starts.tolist(),
                total_counts=counts[:,0].tolist(),right_counts=counts[:,1].tolist(),left_counts=counts[:,2].tolist(),
                unresolved_counts=counts[:,3].tolist(),unresolved_total=unresolved,horizon_ns=horizon,
                bridge_left_hits=int(diagnostics[0]),bridge_right_hits=int(diagnostics[1]),
                bridge_total_hits=int(diagnostics[:2].sum()),max_steps_used=int(diagnostics[2]),
                total_particle_steps=int(diagnostics[3]),bridge_probability_renormalizations=int(diagnostics[4]),
                maximum_bridge_probability_sum=float(maximum[0]),elapsed_seconds=time.monotonic()-t,
                P_FP_nm_ns=control["FP"]["P_nm_ns"],P_ISD_nm_ns=control["ISD"]["P_cm_s"]/100,
                counts_sha256=hashlib.sha256(counts.tobytes()).hexdigest())
    if not unresolved:
        exitcounts=ExitCounts(starts,counts[:,0],counts[:,1],counts[:,2],counts[:,3],int(diagnostics[2]))
        fit=boundary_linear_permeability_from_committor(exitcounts,p[16],-H,max(cfg["offsets_nm"]))
        relative=abs(fit["P_nm_ns"]/result["P_FP_nm_ns"]-1.)
        zscore=abs(fit["P_nm_ns"]-result["P_FP_nm_ns"])/fit["P_SE_nm_ns"]
        result.update(fit,P_BD_nm_ns=fit["P_nm_ns"],P_BD_SE_nm_ns=fit["P_SE_nm_ns"],
                      P_BD_MC95_lower=fit["P_nm_ns"]-1.96*fit["P_SE_nm_ns"],
                      P_BD_MC95_upper=fit["P_nm_ns"]+1.96*fit["P_SE_nm_ns"],
                      relative_BD_vs_FP=relative,zscore_BD_vs_FP=zscore,
                      q_right=(counts[:,1]/counts[:,0]).tolist(),
                      q_jeffreys95_lower=beta.ppf(.025,counts[:,1]+.5,counts[:,2]+.5).tolist(),
                      q_jeffreys95_upper=beta.ppf(.975,counts[:,1]+.5,counts[:,2]+.5).tolist())
        passed=(int(counts[:,1].sum())>=cfg["minimum_right_exits"] and int(diagnostics[:2].sum())>0 and
                relative<=cfg["max_relative_BD_vs_FP"] and zscore<=cfg["max_z_BD_vs_FP"] and
                fit["reduced_chi2"]<=cfg["max_fit_reduced_chi2"])
        result["status"]="PASS" if passed else "INVESTIGATE"
    else:
        result["status"]="NOT_AUDITABLE_CENSORED"
        # Intervalos de committor válidos sem atribuir a direção de saída aos censurados.
        result["q_lower_from_censoring"]=(counts[:,1]/counts[:,0]).tolist()
        result["q_upper_from_censoring"]=((counts[:,1]+counts[:,3])/counts[:,0]).tolist()
    target=ROOT/"evidencias"/f"BD_{control['case_id']}_{spec['method']}_{spec['dt_ns']:.3f}.json"
    write_json(target,result)
    print(f"BD {control['case_id']} {spec['method']} dt={spec['dt_ns']}: {result['status']}; "
          f"direita={int(counts[:,1].sum())}; censura={unresolved}; segundos={result['elapsed_seconds']:.1f}",flush=True)
    return result


def main():
    verify_inputs();controls=prepare();cfg=PROTOCOL["bd"]
    specs=[]
    for case_index,control in enumerate(controls,1):
        for run_index,run_config in enumerate(cfg["runs"],1):
            specs.append(dict(control=control,**run_config,seed=cfg["seed_base"]+10*case_index+run_index))
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,s) for s in specs]
        for future in as_completed(futures):rows.append(future.result())
    rows=sorted(rows,key=lambda x:(x["case_id"],x["method"],x["dt_ns"]))
    comparisons=[]
    for control in controls:
        group=[x for x in rows if x["case_id"]==control["case_id"]]
        primary=next(x for x in group if x["method"]=="EM" and x["dt_ns"]==.002)
        for comparator,limit in [(next(x for x in group if x["method"]=="MILSTEIN"),cfg["max_cross_method_z"]),
                                  (next(x for x in group if x["dt_ns"]==.001),cfg["max_time_refinement_z"])]:
            if all(x["status"]=="PASS" for x in [primary,comparator]):
                zscore=abs(primary["P_BD_nm_ns"]-comparator["P_BD_nm_ns"])/math.hypot(
                    primary["P_BD_SE_nm_ns"],comparator["P_BD_SE_nm_ns"])
                comparisons.append(dict(case_id=control["case_id"],primary_method="EM",primary_dt=.002,
                                        comparator_method=comparator["method"],comparator_dt=comparator["dt_ns"],
                                        zscore=zscore,limit=limit,status="PASS" if zscore<=limit else "INVESTIGATE"))
    margins=[]
    for config in cfg["runs"]:
        group={x["case_id"]:x for x in rows if x["method"]==config["method"] and x["dt_ns"]==config["dt_ns"]}
        if not all(x["status"]=="PASS" for x in group.values()):continue
        logs={key:math.log(x["P_BD_nm_ns"]) for key,x in group.items()}
        variances={key:(x["P_BD_SE_nm_ns"]/x["P_BD_nm_ns"])**2 for key,x in group.items()}
        ag=logs["FLAT"]-logs["G_ONLY"];ad=logs["FLAT"]-logs["D_ONLY"]
        interaction=logs["G_ONLY"]+logs["D_ONLY"]-logs["GD_BOUNDARY"]-logs["FLAT"]
        delta=logs["D_ONLY"]-logs["G_ONLY"]
        se=math.sqrt(variances["D_ONLY"]+variances["G_ONLY"])
        margins.append(dict(**config,AG_BD=ag,AD_BD=ad,I_BD=interaction,deltaC_BD=delta,deltaC_MC_SE=se,
                            deltaC_MC95_lower=delta-1.96*se,deltaC_MC95_upper=delta+1.96*se,
                            zero_inside_MC95=delta-1.96*se<=0<=delta+1.96*se,
                            shared_flat_noise_cancels_in_delta=True,
                            AG_MC_SE=math.sqrt(variances["FLAT"]+variances["G_ONLY"]),
                            AD_MC_SE=math.sqrt(variances["FLAT"]+variances["D_ONLY"]),
                            I_MC_SE=math.sqrt(sum(variances.values()))))
    passed=all(x["status"]=="PASS" for x in rows+comparisons)
    write_json(ROOT/"evidencias/resumo_bd.json",dict(status="PASS" if passed else "REVIEW",run_N=len(rows),
               independent_seed_runs=True,rows=rows,comparisons=comparisons,operational_boundary_margins=margins,
               production_ensemble_covered_by_BD=False,physical_confidence_intervals=False))
    print("Campanha Browniana concluída:","PASS" if passed else "REVIEW",flush=True)


if __name__=="__main__":main()
