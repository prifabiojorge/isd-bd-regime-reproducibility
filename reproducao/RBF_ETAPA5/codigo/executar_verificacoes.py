"""Campanha determinística: malha, margens, quadratura adaptativa e FP direto."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib.metadata
import platform
import time
from numerica import *
from revisao_producao import complete_revision


def integrate_chunk(spec):
    start,theta,n=spec
    return start,integrate_batch(theta,n)


def parallel_integrate(theta,n,workers=3):
    out={key:np.empty(len(theta)) for key in METRICS}
    chunks=[(i,theta[i:i+2048],n) for i in range(0,len(theta),2048)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(integrate_chunk,s) for s in chunks]
        for future in as_completed(futures):
            start,result=future.result()
            for key,value in result.items():out[key][start:start+len(value)]=value
    return out


def error_estimates(coarse,fine):
    floor=PROTOCOL["integration"]["roundoff_floor"]
    result={key:2*np.abs(fine[key]-coarse[key])+floor for key in METRICS if key!="P_cm_s"}
    result["P_relative"]=2*np.abs(fine["P_cm_s"]/coarse["P_cm_s"]-1.)+floor
    result["deltaC"]=result["AG"]+result["AD"]
    return result


def quad_case(spec):
    row_index,case_id,point,theta=spec
    result=adaptive_quad(theta,PROTOCOL["integration"]["quad_epsabs"])
    result.update(row_index=row_index,source_case_id=case_id,point_index=point)
    return result


def fp_case(spec):
    row_index,case_id,point,theta=spec
    rows=[]
    for n in PROTOCOL["fp"]["grids"]+[PROTOCOL["fp"]["audit_highest_grid"]]:
        states={}
        for name in ["FLAT","G_ONLY","D_ONLY","GD"]:
            p=theta.copy()
            if name in ["FLAT","D_ONLY"]:p[:3]=0.
            if name in ["FLAT","G_ONLY"]:p[8:11]=0.
            states[name]=fp_direct(p,n,precision=PROTOCOL["fp"]["decimal_precision"],
                                   floating_diagnostic=name=="GD" and n==6401)
        ag=math.log(states["FLAT"]["P_nm_ns"]/states["G_ONLY"]["P_nm_ns"])
        ad=math.log(states["FLAT"]["P_nm_ns"]/states["D_ONLY"]["P_nm_ns"])
        total=math.log(states["FLAT"]["P_nm_ns"]/states["GD"]["P_nm_ns"])
        interaction=total-ag-ad
        row=dict(row_index=row_index,source_case_id=case_id,point_index=point,grid_N=n,
                 AG=ag,AD=ad,interaction=interaction,CG=ag+interaction/2,CD=ad+interaction/2,
                 deltaC=ag-ad,P_cm_s=states["GD"]["P_cm_s"],
                 max_flux_nonuniformity=max(x["relative_flux_nonuniformity"] for x in states.values()),
                 max_scaled_equation_residual=max(x["maximum_scaled_equation_residual"] for x in states.values()),
                 max_relative_direct_vs_network=max(x["relative_direct_vs_network"] for x in states.values()),
                 max_conductance_ratio=max(x["conductance_ratio"] for x in states.values()),
                 min_density=min(x["p_min"] for x in states.values()),
                 float64_P_relative_error=(abs(states["GD"]["float64_flux_mean"]/states["GD"]["P_nm_ns"]-1.)
                                           if n==6401 and states["GD"].get("float64_error") is None else None))
        if n==6401:
            row["float64_interface_flux_min"]=states["GD"].get("float64_flux_min")
            row["float64_interface_flux_max"]=states["GD"].get("float64_flux_max")
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-meshes",action="store_true",help="Retomar a consolidação com malhas já calculadas e validadas pelas entradas imutáveis.")
    args=parser.parse_args()
    t=time.monotonic();verify_inputs()
    base=pd.read_csv(ROOT/"originais/producao_historica.csv.gz",float_precision="round_trip")
    base["proxy_margin"]=base.g_max.to_numpy()-math.log(10)*base.eta.to_numpy()
    theta=base[FIELDS].to_numpy(float)
    print("Refinando todos os 56.926 aceitos para 6.401 nós...",flush=True)
    saved_all=ROOT/"dados/todos_originais_6401.npz"
    if args.reuse_meshes and saved_all.exists():
        cache=np.load(saved_all,allow_pickle=False)
        assert np.array_equal(cache["sobol_index"],base.sobol_index.to_numpy(int))
        all64={key:cache[key] for key in METRICS}
    else:all64=parallel_integrate(theta,6401)
    np.savez_compressed(ROOT/"dados/todos_originais_6401.npz",sobol_index=base.sobol_index.to_numpy(int),**all64)
    print("Refinamento completo de 56.926 perfis concluído.",flush=True)
    chosen=pd.read_csv(ROOT/"dados/casos_selecionados.csv.gz",float_precision="round_trip")
    p=chosen[FIELDS].to_numpy(float)
    meshes={}
    for n in PROTOCOL["integration"]["targeted_grids"]:
        print(f"Calculando {len(chosen)} perfis selecionados em {n} nós...",flush=True)
        saved=ROOT/"dados"/f"selecionados_{n}.npz"
        if args.reuse_meshes and saved.exists():
            cache=np.load(saved,allow_pickle=False)
            assert np.array_equal(cache["point_index"],chosen.point_index.to_numpy(int))
            assert np.array_equal(cache["source_case_id"],chosen.source_case_id.to_numpy(str))
            meshes[n]={key:cache[key] for key in METRICS}
        else:meshes[n]=parallel_integrate(p,n)
        np.savez_compressed(ROOT/"dados"/f"selecionados_{n}.npz",point_index=chosen.point_index.to_numpy(int),
                            source_case_id=chosen.source_case_id.to_numpy(str),**meshes[n])
    checks=np.unique(np.r_[np.linspace(0,len(chosen)-1,96,dtype=int),
                           np.argsort(np.abs(chosen.deltaC.to_numpy()))[:32]])
    maxerrors={key:0. for key in METRICS}
    for i in checks:
        source=source_metrics(p[i],6401)
        for key in METRICS:
            error=abs(meshes[6401][key][i]/source[key]-1.) if key=="P_cm_s" else abs(meshes[6401][key][i]-source[key])
            maxerrors[key]=max(maxerrors[key],error)
    assert max(maxerrors.values())<2e-12,maxerrors
    write_json(ROOT/"evidencias/validacao_quadratura.json",dict(status="PASS",N=len(checks),
               original_scalar_module_comparator=True,maximum_errors=maxerrors,P_error_relative=True))
    # Toda a produção: incerteza estimada versus registro histórico, com substituições dos refinamentos dirigidos.
    coarse={key:base[key].to_numpy(float) for key in METRICS}
    final={key:all64[key].copy() for key in METRICS};prior={key:coarse[key].copy() for key in METRICS}
    grids=np.full(len(base),6401,dtype=int)
    lookup={int(i):j for j,i in enumerate(base.sobol_index)}
    for i,row in enumerate(chosen.itertuples()):
        if row.source_case_id!="referencia_original":continue
        j=lookup[int(row.point_index)];grids[j]=25601
        for key in METRICS:
            final[key][j]=meshes[25601][key][i];prior[key][j]=meshes[12801][key][i]
    initial_error=error_estimates(coarse,all64)
    errors=error_estimates(prior,final)
    relative_3201_6401=np.abs(all64["P_cm_s"]/base.P_cm_s.to_numpy()-1.)
    tolerance=PROTOCOL["integration"]["contribution_absolute_convergence"]
    bound_crosses=(np.abs(final["deltaC"])<=errors["deltaC"])
    proxy_crosses=(np.abs(final["proxy_margin"])<=errors["proxy_margin"])
    contribution_fail=np.maximum.reduce([errors[k] for k in ["AG","AD","CG","CD","interaction"]])>tolerance
    need=bound_crosses|proxy_crosses|contribution_fail|(errors["P_relative"]>PROTOCOL["integration"]["original_relative_P_convergence"])
    adaptive_rows=[]
    if np.any(need):
        selected=np.flatnonzero(need)
        print(f"Refinamento adaptativo de {len(selected)} perfis originais adicionais.",flush=True)
        low=parallel_integrate(theta[selected],25601);high=parallel_integrate(theta[selected],51201)
        newerr=error_estimates(low,high)
        for pos,j in enumerate(selected):
            grids[j]=51201
            for key in METRICS:final[key][j]=high[key][pos];prior[key][j]=low[key][pos]
            for key in errors:errors[key][j]=newerr[key][pos]
            adaptive_rows.append(dict(sobol_index=int(base.sobol_index.iloc[j]),grid_N=51201,
                                      deltaC=float(final["deltaC"][j]),deltaC_error_estimate=float(errors["deltaC"][j]),
                                      P_relative_error_estimate=float(errors["P_relative"][j])))
    write_json(ROOT/"evidencias/refinamentos_adaptativos.json",dict(N=len(adaptive_rows),rows=adaptive_rows))
    original_delta=base.deltaC.to_numpy();original_m=base.proxy_margin.to_numpy()
    assessment=pd.DataFrame(dict(sobol_index=base.sobol_index.to_numpy(int),source_grid_N=base.evaluation_grid_N,
                                assessment_grid_N=grids,original_deltaC=original_delta,final_deltaC=final["deltaC"],
                                AG_error_estimate=errors["AG"],AD_error_estimate=errors["AD"],
                                deltaC_error_estimate=errors["deltaC"],
                                deltaC_interval_lower=final["deltaC"]-errors["deltaC"],
                                deltaC_interval_upper=final["deltaC"]+errors["deltaC"],
                                interval_reaches_zero=np.abs(final["deltaC"])<=errors["deltaC"],
                                original_proxy_margin=original_m,final_proxy_margin=final["proxy_margin"],
                                proxy_error_estimate=errors["proxy_margin"],
                                proxy_interval_reaches_zero=np.abs(final["proxy_margin"])<=errors["proxy_margin"],
                                original_P_cm_s=base.P_cm_s,final_P_cm_s=final["P_cm_s"],
                                P_relative_error_estimate=errors["P_relative"],
                                relative_P_historical_to_6401=relative_3201_6401,
                                integrated_label_changed=np.sign(original_delta)!=np.sign(final["deltaC"]),
                                proxy_label_changed=np.sign(original_m)!=np.sign(final["proxy_margin"]),
                                boundary_0p10_original=np.abs(original_delta)<=.1))
    assessment.to_csv(ROOT/"dados/precisao_todos_originais.csv.gz",index=False,float_format="%.17g",
                      compression=dict(method="gzip",mtime=0))
    # Não corrigir silenciosamente: registrar a necessidade de revisão do registro de produção.
    requires=(base.evaluation_grid_N.to_numpy()==3201)&(
        (relative_3201_6401>PROTOCOL["integration"]["original_relative_P_convergence"])|
        (np.maximum.reduce([np.abs(all64[k]-coarse[k]) for k in ["AG","AD","CG","CD","interaction"]])>tolerance))
    revised=base.copy()
    for key in METRICS:revised.loc[requires,key]=final[key][requires]
    revised["stage5_revision_applied"]=requires
    revised["stage5_assessment_grid_N"]=grids
    # Ajuste dos descritores dependentes se houver revisão necessária.
    if np.any(requires):
        revised.loc[requires,"eta"]=-np.log10(revised.loc[requires,"d_min"])
        revised.loc[requires,"wG"]=np.exp(revised.loc[requires,"AG"]-revised.loc[requires,"g_max"])
        revised.loc[requires,"wD"]=np.exp(revised.loc[requires,"AD"]+np.log(revised.loc[requires,"d_min"]))
        revised.loc[requires,"log10_P_cm_s"]=np.log10(revised.loc[requires,"P_cm_s"])
        revised.loc[requires,"evaluation_grid_N"]=grids[requires]
        revised.loc[requires,"numerical_refinement_applied"]=True
        complete_revision(revised,theta,grids,requires)
    revised.to_csv(ROOT/"dados/producao_revisao_numerica.csv.gz",index=False,float_format="%.17g",
                   compression=dict(method="gzip",mtime=0))
    # Estabilidade dos casos adicionais; o conjunto original já está coberto integralmente acima.
    selected_errors=error_estimates(meshes[12801],meshes[25601])
    selected_summary=[]
    for case,group in chosen.groupby("source_case_id"):
        ix=group.index.to_numpy(int);old=group.deltaC.to_numpy()
        selected_summary.append(dict(case_id=case,N=len(ix),
            delta_interval_zero_N=int(np.sum(np.abs(meshes[25601]["deltaC"][ix])<=selected_errors["deltaC"][ix])),
            integrated_label_changed_N=int(np.sum(np.sign(old)!=np.sign(meshes[25601]["deltaC"][ix]))),
            proxy_label_changed_N=int(np.sum(np.sign(group.proxy_margin)!=np.sign(meshes[25601]["proxy_margin"][ix]))),
            max_delta_error_estimate=float(selected_errors["deltaC"][ix].max()),
            minimum_abs_delta=float(np.abs(meshes[25601]["deltaC"][ix]).min()),
            max_P_relative_error_estimate=float(selected_errors["P_relative"][ix].max())))
    pd.DataFrame(selected_summary).to_csv(ROOT/"dados/precisao_por_cenario.csv",index=False,float_format="%.17g")
    # Pontos com resíduo pequeno em Richardson são rotulados como dominados por arredondamento.
    convergence=[]
    for key in ["AG","AD","CG","CD","interaction","deltaC","P_cm_s"]:
        e1=np.abs(meshes[6401][key]-meshes[3201][key]);e2=np.abs(meshes[12801][key]-meshes[6401][key])
        mask=(e1>1e-11)&(e2>1e-12)
        ratios=e1[mask]/e2[mask]
        convergence.append(dict(field=key,N_resolved_order=int(mask.sum()),
             ratio_median=float(np.median(ratios)) if len(ratios) else None,
             ratio_Q05=float(np.quantile(ratios,.05)) if len(ratios) else None,
             ratio_Q95=float(np.quantile(ratios,.95)) if len(ratios) else None,
             max_difference_3201_6401=float(e1.max()),max_difference_6401_12801=float(e2.max())))
    pd.DataFrame(convergence).to_csv(ROOT/"dados/convergencia_por_grandeza.csv",index=False,float_format="%.17g")
    np.savez_compressed(ROOT/"dados/avaliacao_final_original.npz",sobol_index=base.sobol_index.to_numpy(int),
                         assessment_grid_N=grids,**final)
    write_json(ROOT/"evidencias/resumo_integracao.json",dict(status="PASS" if not assessment.interval_reaches_zero.any() else "REVIEW",
          original_N=len(base),selected_N=len(chosen),selected_original_N=int((chosen.source_case_id=="referencia_original").sum()),
          original_boundary_985_N=int(assessment.boundary_0p10_original.sum()),
          original_integrated_interval_zero_N=int(assessment.interval_reaches_zero.sum()),
          original_proxy_interval_zero_N=int(assessment.proxy_interval_reaches_zero.sum()),
          original_integrated_label_changed_N=int(assessment.integrated_label_changed.sum()),
          original_proxy_label_changed_N=int(assessment.proxy_label_changed.sum()),
          max_delta_error_estimate=float(errors["deltaC"].max()),max_P_error_estimate=float(errors["P_relative"].max()),
          minimum_abs_original_delta=float(np.abs(final["deltaC"]).min()),
          production_revision_required_N=int(requires.sum()),revision_sobol_indices=base.loc[requires,"sobol_index"].tolist(),
          adaptive_N=len(adaptive_rows),max_errors_by_field={key:float(value.max()) for key,value in errors.items()},
          selected_by_case=selected_summary,convergence=convergence,
          fine_grid_filter_gmin_below_N=int(np.sum(final["g_min"]<-22.34161393982527-2e-12)),
          fine_grid_filter_gmax_above_N=int(np.sum(final["g_max"]>13.03260813156474+2e-12)),
          fine_grid_filter_dmin_below_N=int(np.sum(final["d_min"]<.09-2e-12)),
          fine_grid_filter_dmax_above_N=int(np.sum(final["d_max"]>1+2e-12)),
          measured_error_intervals_not_certified_bounds=True))
    print("Integração e fronteira concluídas.",flush=True)
    # Quadratura adaptativa independente da malha uniforme.
    specs=[(i,r.source_case_id,int(r.point_index),p[i]) for i,r in enumerate(chosen.itertuples()) if r.quad_selected]
    quadrows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(quad_case,specs):
            i=row["row_index"]
            for key in ["AG","AD","interaction","CG","CD","deltaC"]:
                row["abs_uniform25601_vs_quad_"+key]=abs(meshes[25601][key][i]-row[key])
            row["relative_uniform25601_vs_quad_P"]=abs(meshes[25601]["P_cm_s"][i]/row["P_cm_s"]-1.)
            row["uniform_error_estimate_delta"]=float(selected_errors["deltaC"][i])
            quadrows.append(row)
    write_json(ROOT/"evidencias/quadpack.json",dict(rows=quadrows,N=len(quadrows),
          max_abs_delta_difference=max(x["abs_uniform25601_vs_quad_deltaC"] for x in quadrows),
          max_relative_P_difference=max(x["relative_uniform25601_vs_quad_P"] for x in quadrows),
          quad_error_estimates_are_not_rigorous_bounds=True))
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,list)} for x in quadrows]).to_csv(
        ROOT/"dados/quadpack_comparacao.csv",index=False,float_format="%.17g")
    print("Comparação QUADPACK concluída.",flush=True)
    # FP em quatro estados, garantindo comparação das contribuições e da permeabilidade.
    specs=[(i,r.source_case_id,int(r.point_index),p[i]) for i,r in enumerate(chosen.itertuples()) if r.fp_selected]
    fprows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(fp_case,s) for s in specs]
        for k,future in enumerate(as_completed(futures),1):
            fprows.extend(future.result())
            if k%24==0 or k==len(specs):print(f"FP direto: {k}/{len(specs)} perfis, quatro estados e quatro malhas.",flush=True)
    fprows.sort(key=lambda x:(x["source_case_id"],x["point_index"],x["grid_N"]))
    for row in fprows:
        i=row["row_index"];n=row["grid_N"]
        row["relative_FP_vs_ISD"]=abs(row["P_cm_s"]/meshes[n]["P_cm_s"][i]-1.)
        row["abs_FP_vs_ISD_deltaC"]=abs(row["deltaC"]-meshes[n]["deltaC"][i])
        row["integrated_sign_differs_from_ISD"]=int(np.sign(row["deltaC"]))!=int(np.sign(meshes[n]["deltaC"][i]))
    fpframe=pd.DataFrame(fprows)
    fpframe.to_csv(ROOT/"dados/fokker_planck_direto.csv",index=False,float_format="%.17g")
    fp_summary=[]
    for n,group in fpframe.groupby("grid_N"):
        fp_summary.append(dict(grid_N=int(n),N=len(group),max_relative_P=float(group.relative_FP_vs_ISD.max()),
                               max_abs_delta=float(group.abs_FP_vs_ISD_deltaC.max()),
                               sign_difference_N=int(group.integrated_sign_differs_from_ISD.sum())))
    cfg=PROTOCOL["fp"]
    passed=(fpframe.max_flux_nonuniformity.max()<=cfg["max_flux_nonuniformity"] and
            fpframe.max_scaled_equation_residual.max()<=cfg["max_equation_backward_residual"] and
            fpframe.max_relative_direct_vs_network.max()<=cfg["max_relative_direct_vs_network"] and
            fpframe[fpframe.grid_N>=6401].relative_FP_vs_ISD.max()<=cfg["max_relative_FP_vs_ISD"])
    write_json(ROOT/"evidencias/resumo_fp.json",dict(status="PASS" if passed else "REVIEW",profile_N=len(specs),
               counterfactual_states_per_profile=4,grids=cfg["grids"]+[cfg["audit_highest_grid"]],rows=fp_summary,
               max_flux_nonuniformity=float(fpframe.max_flux_nonuniformity.max()),
               max_scaled_equation_residual=float(fpframe.max_scaled_equation_residual.max()),
               max_direct_vs_network=float(fpframe.max_relative_direct_vs_network.max()),
               largest_float64_diagnostic_relative_P_error=float(fpframe.float64_P_relative_error.max()),
               float64_diagnostic_not_used_for_production=True,independent_physical_validation=False,
               direct_linear_system_solved=True,history_network_not_claimed_to_have_measured_local_flux_residuals=True))
    packages={name:importlib.metadata.version(name) for name in ["numpy","scipy","pandas","matplotlib","PyYAML","Pillow"]}
    write_json(ROOT/"evidencias/ambiente.json",dict(python=platform.python_version(),platform=platform.platform(),
               packages=packages,decimal_from_standard_library=True,workers=3,elapsed_seconds=time.monotonic()-t))
    (ROOT/"requirements.txt").write_text("\n".join(f"{key}=={value}" for key,value in packages.items())+"\n")
    print("Campanha determinística concluída:","PASS" if passed else "REVIEW",flush=True)


if __name__=="__main__":main()
