"""Reanalisa resultados preservados sem repetir trajetórias nem solvers FP."""
from numerica import *
from isdbd_regime.bd_flux import ExitCounts, boundary_linear_permeability_from_committor


def main():
    verify_inputs()
    load=lambda name:json.loads((ROOT/"evidencias"/name).read_text())
    integration=load("resumo_integracao.json")
    fp_summary=load("resumo_fp.json")
    bd=load("resumo_bd.json")
    quad=load("quadpack.json")
    revision=load("revisao_producao_integridade.json")
    base=pd.read_csv(ROOT/"originais/producao_historica.csv.gz",float_precision="round_trip")
    rev=pd.read_csv(ROOT/"dados/producao_revisao_numerica.csv.gz",float_precision="round_trip")
    precision=pd.read_csv(ROOT/"dados/precisao_todos_originais.csv.gz",float_precision="round_trip")
    required=rev.stage5_revision_applied.to_numpy(bool)
    cols=["P_cm_s","AG","AD","interaction","CG","CD","deltaC"]
    revisions=[]
    all64=np.load(ROOT/"dados/todos_originais_6401.npz",allow_pickle=False)
    for j in np.flatnonzero(required):
        item=dict(sobol_index=int(base.sobol_index.iloc[j]),old_grid_N=int(base.evaluation_grid_N.iloc[j]),
                  revised_grid_N=int(rev.evaluation_grid_N.iloc[j]),
                  original_P_relative_6401=float(abs(all64["P_cm_s"][j]/base.P_cm_s.iloc[j]-1)),
                  original_max_contribution_change_6401=float(max(abs(all64[k][j]-base[k].iloc[j])
                                                for k in ["AG","AD","interaction","CG","CD"])),
                  revised_P_relative_change=float(abs(rev.P_cm_s.iloc[j]/base.P_cm_s.iloc[j]-1)))
        for k in cols:
            item["old_"+k]=float(base[k].iloc[j]);item["revised_"+k]=float(rev[k].iloc[j])
        revisions.append(item)
    pd.DataFrame(revisions).to_csv(ROOT/"dados/revisao_seis_registros.csv",index=False,float_format="%.17g")
    assert not np.any(np.sign(base.deltaC)!=np.sign(rev.deltaC))
    old_m=base.g_max-math.log(10)*base.eta
    new_m=rev.g_max-math.log(10)*rev.eta
    assert not np.any(np.sign(old_m)!=np.sign(new_m))
    assert np.allclose(rev.P_cm_s,100*rev.P_nm_ns,rtol=3e-14,atol=0)
    assert np.allclose(rev.AG,rev.logRG-rev.logR0,atol=3e-14,rtol=0)
    assert np.allclose(rev.AD,rev.logRD-rev.logR0,atol=3e-14,rtol=0)
    assert np.allclose(rev.interaction,rev.logRGD-rev.logRG-rev.logRD+rev.logR0,atol=3e-14,rtol=0)
    counts={"T_to_T":int(((rev.deltaC>0)&(new_m>0)).sum()),
            "T_to_D":int(((rev.deltaC>0)&(new_m<0)).sum()),
            "D_to_T":int(((rev.deltaC<0)&(new_m>0)).sum()),
            "D_to_D":int(((rev.deltaC<0)&(new_m<0)).sum())}
    assert counts==dict(T_to_T=29161,T_to_D=7,D_to_T=4366,D_to_D=23392)
    # Estimativa de discretização do FP obtida com seu próprio refinamento.
    fp=pd.read_csv(ROOT/"dados/fokker_planck_direto.csv",float_precision="round_trip")
    idcols=["source_case_id","point_index"]
    low=fp[fp.grid_N==12801].set_index(idcols)
    high=fp[fp.grid_N==25601].set_index(idcols)
    estimate=2*(abs(high.AG-low.AG)+abs(high.AD-low.AD))+2e-12
    fp_precision=high[["AG","AD","deltaC","P_cm_s"]].copy()
    fp_precision["deltaC_error_estimate"]=estimate
    fp_precision["deltaC_interval_lower"]=high.deltaC-estimate
    fp_precision["deltaC_interval_upper"]=high.deltaC+estimate
    fp_precision["interval_reaches_zero"]=abs(high.deltaC)<=estimate
    fp_precision["P_relative_error_estimate"]=2*abs(high.P_cm_s/low.P_cm_s-1)+1e-12
    fp_precision.reset_index().to_csv(ROOT/"dados/precisao_fokker_planck.csv",index=False,float_format="%.17g")
    fp_aux=dict(N=len(high),interval_reaches_zero_N=int(fp_precision.interval_reaches_zero.sum()),
                maximum_delta_error_estimate=float(estimate.max()),
                minimum_delta_to_error_ratio=float((abs(high.deltaC)/estimate).min()),
                maximum_relative_P_error_estimate=float(fp_precision.P_relative_error_estimate.max()))
    # Reanálise aritmética das contagens históricas. Não reconstitui perfis/seeds ausentes.
    historical=[]
    for filename,key,case in [
        ("r2c2bfix3_bridge_boundary_calibration.json","runs","FLAT"),
        ("r2c2bfix4_moderate_bridge_dt_audit.json","rows","MODERATE")]:
        original=json.loads((ROOT/"originais/controles_bd"/filename).read_text())
        for row in original[key]:
            n=int(row["paths_per_start"]);right=np.array(row["right_counts"],np.int64)
            assert row["unresolved_total"]==0
            starts=np.array(original["offsets_nm"])-H
            c=ExitCounts(starts,np.full(len(starts),n),right,n-right,np.zeros(len(starts),np.int64),0)
            # Dbulk=1 para comparar o ajuste e obter a escala implícita dos registros.
            fit=boundary_linear_permeability_from_committor(c,1.,-H,max(original["offsets_nm"]))
            implied_bulk=row["P_BD_nm_ns"]/fit["P_nm_ns"]
            ratio_error=abs(row["P_BD_nm_ns"]/row["P_BD_SE_nm_ns"]-
                            fit["P_nm_ns"]/fit["P_SE_nm_ns"])
            chi_error=abs(row["reduced_chi2"]-fit["reduced_chi2"])
            assert max(ratio_error,chi_error)<1e-11
            historical.append(dict(case_id=case,method=row.get("method",row.get("scheme")),
                dt_ns=row["dt_ns"],paths_per_start=n,right_total=int(right.sum()),
                unresolved_total=0,implied_D_bulk_nm2_ns=implied_bulk,
                P_over_SE_recalculation_error=ratio_error,reduced_chi2_recalculation_error=chi_error,
                count_based_arithmetic_verified=True,original_trajectories_rerun=False,
                original_profiles_and_seeds_fully_reconstructed=False))
    pd.DataFrame(historical).to_csv(ROOT/"dados/reanalise_bd_historico.csv",index=False,float_format="%.17g")
    bdcols=["case_id","method","dt_ns","seed","paths_per_start","horizon_ns",
            "P_BD_nm_ns","P_BD_SE_nm_ns","P_BD_MC95_lower","P_BD_MC95_upper",
            "P_FP_nm_ns","relative_BD_vs_FP","zscore_BD_vs_FP","reduced_chi2",
            "unresolved_total","bridge_total_hits","status"]
    pd.DataFrame([{**{k:r[k] for k in bdcols},"total_paths":sum(r["total_counts"]),
                   "right_events":sum(r["right_counts"]),"minimum_right_per_start":min(r["right_counts"])}
                  for r in bd["rows"]]).to_csv(ROOT/"dados/brownian_novo_resumo.csv",index=False,float_format="%.17g")
    detailed_margins=[]
    for row in bd["operational_boundary_margins"]:
        item=row.copy()
        item["I_MC95_lower"]=row["I_BD"]-1.96*row["I_MC_SE"]
        item["I_MC95_upper"]=row["I_BD"]+1.96*row["I_MC_SE"]
        for state in ["G","D"]:
            item[f"C{state}_BD"]=row[f"A{state}_BD"]+row["I_BD"]/2
            item[f"C{state}_MC_SE"]=row["I_MC_SE"]/2
            item[f"C{state}_MC95_lower"]=item[f"C{state}_BD"]-1.96*item[f"C{state}_MC_SE"]
            item[f"C{state}_MC95_upper"]=item[f"C{state}_BD"]+1.96*item[f"C{state}_MC_SE"]
        detailed_margins.append(item)
    pd.DataFrame(detailed_margins).to_csv(ROOT/"dados/brownian_margem_fronteira.csv",index=False,float_format="%.17g")
    selected_delta={k:np.load(ROOT/"dados"/f"selecionados_{k}.npz",allow_pickle=False) for k in [3201,6401,12801,25601]}
    quad_meshes=[]
    for n,cache in selected_delta.items():
        rel=[];deltas=[]
        for q in quad["rows"]:
            i=q["row_index"];rel.append(abs(cache["P_cm_s"][i]/q["P_cm_s"]-1))
            deltas.append(abs(cache["deltaC"][i]-q["deltaC"]))
        quad_meshes.append(dict(grid_N=n,N=len(rel),max_relative_P_vs_QUAD=float(max(rel)),
                                max_absolute_delta_vs_QUAD=float(max(deltas))))
    pd.DataFrame(quad_meshes).to_csv(ROOT/"dados/convergencia_quadpack.csv",index=False,float_format="%.17g")
    summary=dict(status="PASS",original_matrix=counts,production_revision_N=int(required.sum()),
      maximum_P_relative_change_revised_records=float(max(r["revised_P_relative_change"] for r in revisions)),
      minimum_original_delta_to_error_ratio=float((abs(precision.final_deltaC)/precision.deltaC_error_estimate).min()),
      original_assessment_grid_counts={str(k):int(v) for k,v in precision.assessment_grid_N.value_counts().sort_index().items()},
      boundary985_minimum_delta_to_error_ratio=float((abs(precision.loc[precision.boundary_0p10_original,"final_deltaC"])/
                          precision.loc[precision.boundary_0p10_original,"deltaC_error_estimate"]).min()),
      fp_discretization_estimates=fp_aux,
      quad_meshes=quad_meshes,quad_no_failure_messages=not any(v for q in quad["rows"] for v in q["quad_messages"]),
      maximum_quad_estimated_delta_error=max(q["quad_delta_error_estimate"] for q in quad["rows"]),
      new_BD_total_paths=int(sum(sum(r["total_counts"]) for r in bd["rows"])),
      new_BD_total_unresolved=int(sum(r["unresolved_total"] for r in bd["rows"])),
      new_BD_minimum_right_per_start=min(min(r["right_counts"]) for r in bd["rows"]),
      new_BD_maximum_relative_difference=max(r["relative_BD_vs_FP"] for r in bd["rows"]),
      new_BD_maximum_z=max(r["zscore_BD_vs_FP"] for r in bd["rows"]),
      new_BD_maximum_cross_control_z=max(r["zscore"] for r in bd["comparisons"]),
      new_BD_maximum_reduced_chi2=max(r["reduced_chi2"] for r in bd["rows"]),
      new_BD_FP_inside_MC95_N=sum(r["P_BD_MC95_lower"]<=r["P_FP_nm_ns"]<=r["P_BD_MC95_upper"] for r in bd["rows"]),
      new_BD_I_MC95_reaches_zero_N=sum(r["I_MC95_lower"]<=0<=r["I_MC95_upper"] for r in detailed_margins),
      historical_BD_count_reanalysis_N=len(historical),
      unresolved_historical_P4_retained=True,FP_and_BD_cover_full_ensemble=False,
      physical_uncertainty_quantified=False,stage6_and_stage7_started=False)
    assert integration["status"]==fp_summary["status"]==bd["status"]==revision["status"]=="PASS"
    assert not any(r["delta_interval_zero_N"] or r["integrated_label_changed_N"] or r["proxy_label_changed_N"]
                   for r in integration["selected_by_case"])
    assert summary["new_BD_total_unresolved"]==0 and fp_aux["interval_reaches_zero_N"]==0
    write_json(ROOT/"evidencias/consolidacao.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
