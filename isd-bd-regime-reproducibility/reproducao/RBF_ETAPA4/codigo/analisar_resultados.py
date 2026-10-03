"""Matrizes, denominadores, suportes e comparações do protocolo executado."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from motor_sensibilidade import *

QUANTILES=[0,.05,.25,.5,.75,.95,1]
SCORES=[-1,0,1]
NAMES={-1:"difusivo",0:"empate",1:"termodinamico"}
XEDGES=np.linspace(0,LIMITS["g_max"]+TAU,14)
YEDGES=np.linspace(0,1.05,15)


def pct(n,d):
    return 100*int(n)/int(d) if d else None


def matrix_and_rates(out,mask):
    accepted=np.asarray(mask)&out["accepted"]
    y,p=out["integrated_class"][accepted],out["proxy_class"][accepted]
    matrix={(a,b):int(np.sum((y==a)&(p==b))) for a in SCORES for b in SCORES}
    n=int(accepted.sum());nt=int(np.sum(y==1));nd=int(np.sum(y==-1))
    pt=int(np.sum(p==1));pd_=int(np.sum(p==-1))
    errt=nt-matrix[(1,1)];errd=nd-matrix[(-1,-1)]
    wrong=int(np.sum(y!=p))
    return matrix,dict(
        accepted_N=n,thermo_N=nt,diffusive_N=nd,integrated_tie_N=int(np.sum(y==0)),
        proxy_thermo_N=pt,proxy_diffusive_N=pd_,proxy_tie_N=int(np.sum(p==0)),
        correct_N=n-wrong,mismatch_N=wrong,
        thermo_as_diffusive_N=matrix[(1,-1)],diffusive_as_thermo_N=matrix[(-1,1)],
        thermo_error_N=errt,diffusive_error_N=errd,
        accuracy_pct=pct(n-wrong,n),mismatch_pct=pct(wrong,n),
        thermo_fraction_pct=pct(nt,n),diffusive_fraction_pct=pct(nd,n),
        thermo_error_pct=pct(errt,nt),diffusive_error_pct=pct(errd,nd),
        wrong_within_proxy_thermo_N=pt-matrix[(1,1)],
        wrong_within_proxy_diffusive_N=pd_-matrix[(-1,-1)],
        wrong_within_proxy_thermo_pct=pct(pt-matrix[(1,1)],pt),
        wrong_within_proxy_diffusive_pct=pct(pd_-matrix[(-1,-1)],pd_),
        AG_negative_N=int(np.sum(out["AG"][accepted]<0)),
        AG_negative_pct=pct(np.sum(out["AG"][accepted]<0),n),
        near_abs_delta_le_0p10_N=int(np.sum(np.abs(out["deltaC"][accepted])<=.1)),
        minimum_abs_delta=float(np.min(np.abs(out["deltaC"][accepted]))),
        minimum_abs_proxy=float(np.min(np.abs(out["proxy_margin"][accepted]))),
    )


def qrows(case_id,kind,out,mask):
    rows=[]
    for field in ["AG","AD","deltaC","rho","proxy_margin","g_max","eta","log10_P_cm_s"]:
        vals=out[field][mask]
        for quant,value in zip(QUANTILES,np.quantile(vals,QUANTILES)):
            rows.append(dict(case_id=case_id,kind=kind,field=field,quantile=quant,value=float(value),N=len(vals)))
    return rows


def error_regions(case_id,kind,out,mask):
    rows=[]
    for name,actual,predicted in [("difusivo_previsto_termodinamico",-1,1),("termodinamico_previsto_difusivo",1,-1)]:
        selected=mask&(out["integrated_class"]==actual)&(out["proxy_class"]==predicted)
        n=int(selected.sum())
        counts=np.histogram2d(out["g_max"][selected],out["eta"][selected],bins=[XEDGES,YEDGES])[0]
        assert int(counts.sum())==n
        row=dict(case_id=case_id,kind=kind,direction=name,N=n,
                 occupied_fixed_cells=int(np.sum(counts>0)),fixed_cells_total=13*14)
        for field in ["g_max","eta","deltaC","rho","proxy_margin"]:
            row[field+"_min"]=float(out[field][selected].min()) if n else None
            row[field+"_max"]=float(out[field][selected].max()) if n else None
        rows.append(row)
    return rows


def joined(a,b):
    assert set(a)==set(b)
    return {k:np.concatenate([a[k],b[k]]) for k in a}


def main():
    base=load_case("referencia_original")
    n0=len(base["point_index"])
    assert n0==65536
    _,u,p,_=points(dict(id="base",m=16,scramble=False))
    domain_rows=[];crop_rows=[];sampling_rows=[];matrices=[];quantiles=[];regions=[]
    all_summaries=[]

    def add(case_id,label,kind,out,eligible=None,draw_N=None):
        if eligible is None:eligible=np.ones(len(out["accepted"]),dtype=bool)
        eligible=np.asarray(eligible,dtype=bool)
        mask=eligible&out["accepted"]
        mat,rates=matrix_and_rates(out,eligible)
        row=dict(case_id=case_id,label=label,kind=kind,original_draw_N=draw_N or len(eligible),
                 candidate_N=int(eligible.sum()),rejected_N=int(eligible.sum())-rates["accepted_N"],
                 acceptance_pct=pct(rates["accepted_N"],eligible.sum()),**rates)
        for bit,name in [(1,"g_lower"),(2,"g_upper"),(4,"d_lower"),(8,"d_upper")]:
            row["fails_nonexclusive_"+name]=int(np.sum(eligible&((out["fail_bits"]&bit)!=0)))
        all_summaries.append(row)
        for (a,b),n in mat.items():
            matrices.append(dict(case_id=case_id,kind=kind,actual=NAMES[a],proxy=NAMES[b],N=n,
                                 denominator_actual=int(np.sum(mask&(out["integrated_class"]==a))),
                                 denominator_proxy=int(np.sum(mask&(out["proxy_class"]==b)))))
        quantiles.extend(qrows(case_id,kind,out,mask))
        regions.extend(error_regions(case_id,kind,out,mask))
        return row

    baseline=add("referencia_original","Referência original","historical_reference",base)
    assert baseline["accepted_N"]==56926 and baseline["thermo_N"]==29168 and baseline["diffusive_N"]==27758
    assert baseline["mismatch_N"]==4373 and baseline["diffusive_as_thermo_N"]==4366 and baseline["thermo_as_diffusive_N"]==7
    assert baseline["integrated_tie_N"]==baseline["proxy_tie_N"]==0
    domain_rows.append(baseline)
    pairs=[];bounds_rows=[]
    for spec in PROTOCOL["domain_scenarios"]:
        case=load_case(spec["id"])
        row=add(spec["id"],spec["label"],"domain_remap",case)
        domain_rows.append(row)
        lo,hi=scenario_bounds(spec)
        qualifies=np.all((p>=lo)&(p<hi),axis=1)
        crop=add("recorte_"+spec["id"],"Recorte: "+spec["label"],"posthoc_parameter_restriction",base,qualifies,65536)
        crop_rows.append(crop)
        common=base["accepted"]&case["accepted"]
        bm=base["integrated_class"]!=base["proxy_class"]
        cm=case["integrated_class"]!=case["proxy_class"]
        transitions=[]
        for a in [-1,1]:
            for b in [-1,1]:
                transitions.append(dict(before=NAMES[a],after=NAMES[b],N=int(np.sum(common&(base["integrated_class"]==a)&(case["integrated_class"]==b)))))
        pairs.append(dict(case_id=spec["id"],candidate_N=65536,common_accepted_N=int(common.sum()),
                          only_original_accepted_N=int(np.sum(base["accepted"]&~case["accepted"])),
                          only_new_accepted_N=int(np.sum(~base["accepted"]&case["accepted"])),
                          integrated_label_changed_N=int(np.sum(common&(base["integrated_class"]!=case["integrated_class"]))),
                          proxy_label_changed_N=int(np.sum(common&(base["proxy_class"]!=case["proxy_class"]))),
                          mismatches_original_common_N=int(np.sum(common&bm)),mismatches_new_common_N=int(np.sum(common&cm)),
                          correct_to_wrong_N=int(np.sum(common&~bm&cm)),wrong_to_correct_N=int(np.sum(common&bm&~cm)),
                          transitions=transitions))
        for i,field in enumerate(FIELDS):
            bounds_rows.append(dict(case_id=spec["id"],field=field,unit=BOUNDS.unit.iloc[i],
                                    original_low=float(LO[i]),original_high=float(HI[i]),
                                    scenario_low=float(lo[i]),scenario_high=float(hi[i]),changed=bool(lo[i]!=LO[i] or hi[i]!=HI[i])))

    cuts={}
    for threshold in PROTOCOL["posthoc_d_min"]:
        cut={k:v.copy() for k,v in base.items()}
        bad=cut["filter_d_min"]<threshold-TAU
        cut["fail_bits"][bad]|=4
        cut["accepted"]=cut["fail_bits"]==0
        assert np.all(~cut["accepted"]|base["accepted"])
        cut["integrated_class"][~cut["accepted"]]=9;cut["proxy_class"][~cut["accepted"]]=9
        case_id="recorte_dmin_"+str(threshold).replace(".","p")
        row=add(case_id,f"Recorte d ≥ {threshold:.2f}","posthoc_filter_restriction",cut)
        domain_rows.append(row)
        cuts[case_id]=cut["accepted"]
    np.savez_compressed(ROOT/"dados/recortes_supressao.npz",point_index=base["point_index"],**cuts)

    extension=load_case("extensao_sobol_17")
    assert extension["point_index"][0]==65536 and extension["point_index"][-1]==131071
    combined=joined(base,extension)
    for m in PROTOCOL["sampling_prefix_m"]:
        n=2**m;out=combined if m==17 else base
        row=add(f"prefixo_sobol_{m}",f"Sobol não embaralhado: 2^{m}","sampling_nested",out,out["point_index"]<n,n)
        sampling_rows.append(row)
    for i,seed in enumerate(PROTOCOL["scrambled_seed"],1):
        out=load_case(f"sobol_embaralhado_{i}")
        row=add(f"sobol_embaralhado_{i}",f"Sobol embaralhado {i}","sampling_scramble",out)
        row["seed"]=int(seed);sampling_rows.append(row)

    cohorts=[]
    cohort_masks=[("AG_negativo",base["AG"]<0),("AG_nao_negativo",base["AG"]>=0),("g_sem_pocos",base["g_min"]>=0)]
    for case_id,eligible in cohort_masks:
        mat,rates=matrix_and_rates(base,eligible)
        cohorts.append(dict(cohort=case_id,**rates))
        for (a,b),n in mat.items():
            matrices.append(dict(case_id=case_id,kind="accepted_profile_conditioning",actual=NAMES[a],proxy=NAMES[b],N=n,
                                 denominator_actual=int(np.sum(base["accepted"]&eligible&(base["integrated_class"]==a))),
                                 denominator_proxy=int(np.sum(base["accepted"]&eligible&(base["proxy_class"]==b)))))
    wrongd=base["accepted"]&(base["integrated_class"]==-1)&(base["proxy_class"]==1)
    wrongt=base["accepted"]&(base["integrated_class"]==1)&(base["proxy_class"]==-1)
    assert np.all(base["rho"][wrongd]<-base["proxy_margin"][wrongd])
    assert np.all(base["rho"][wrongt]>-base["proxy_margin"][wrongt])
    groups=[("T_previsto_T",base["accepted"]&(base["integrated_class"]==1)&(base["proxy_class"]==1)),
            ("D_previsto_D",base["accepted"]&(base["integrated_class"]==-1)&(base["proxy_class"]==-1)),
            ("D_previsto_T",wrongd),("T_previsto_D",wrongt)]
    support_rows=[];parameter_quant=[]
    for name,mask in groups:
        row=dict(group=name,N=int(mask.sum()),AG_negative_N=int(np.sum(mask&(base["AG"]<0))),
                 median_rho=float(np.median(base["rho"][mask])),
                 median_wG_over_wD=float(np.median(base["wG"][mask]/base["wD"][mask])),
                 median_proxy_margin=float(np.median(base["proxy_margin"][mask])),
                 median_deltaC=float(np.median(base["deltaC"][mask])),
                 median_wG=float(np.median(base["wG"][mask])),median_wD=float(np.median(base["wD"][mask])))
        for field in ["g_min","g_max","d_min","eta","AG","AD"]:
            row[field+"_median"]=float(np.median(base[field][mask]))
        support_rows.append(row)
        for j,field in enumerate(FIELDS):
            for q,value in zip(QUANTILES,np.quantile(p[mask,j],QUANTILES)):
                parameter_quant.append(dict(group=name,N=int(mask.sum()),field=field,quantile=q,value=float(value)))

    scramble_df=pd.DataFrame([r for r in sampling_rows if r["kind"]=="sampling_scramble"])
    sampling_summary={}
    for key in ["acceptance_pct","thermo_fraction_pct","mismatch_pct","thermo_error_pct","diffusive_error_pct",
                "thermo_as_diffusive_N","diffusive_as_thermo_N"]:
        vals=scramble_df[key].to_numpy(float)
        sampling_summary[key]=dict(min=float(vals.min()),max=float(vals.max()),mean=float(vals.mean()),
                                   sample_sd=float(vals.std(ddof=1)),replicates=len(vals),
                                   meaning="Estatística descritiva de quatro desenhos sintéticos, sem IC populacional")
    write_json(ROOT/"evidencias/analise_completa.json",dict(
        status="PASS",baseline=baseline,domain=domain_rows,parameter_restrictions=crop_rows,sampling=sampling_rows,
        paired_comparisons=pairs,AG_cohorts=cohorts,support_groups=support_rows,
        scramble_summary=sampling_summary,
        AG_negative_in_diffusive_proxy_thermo_N=int(np.sum(wrongd&(base["AG"]<0))),
        AG_negative_share_of_diffusive_proxy_thermo_pct=pct(np.sum(wrongd&(base["AG"]<0)),wrongd.sum()),
        structural_AD_nonnegative_verified=True,shape_identity_verified_in_every_case=True,
        first_prefix_unchanged=True,physical_population_interpretation=False,
        fixed_region_grid=dict(x_edges=XEDGES.tolist(),eta_edges=YEDGES.tolist(),
                               interpretation="Células com pontos discordantes; não estimativa de área contínua ou prevalência"),
    ))
    tables={"resumo_dominio":domain_rows,"recortes_parametricos":crop_rows,"resumo_amostragem":sampling_rows,
            "todas_matrizes":matrices,"quantis_contribuicoes":quantiles,"regioes_discordancia":regions,
            "coortes_AG":cohorts,"suportes_por_grupo":support_rows,"quantis_parametros_por_grupo":parameter_quant,
            "limites_cenarios":bounds_rows}
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(ROOT/"dados"/(name+".csv"),index=False,float_format="%.17g")
    simple_pairs=[{k:v for k,v in r.items() if k!="transitions"} for r in pairs]
    pd.DataFrame(simple_pairs).to_csv(ROOT/"dados/comparacoes_pareadas.csv",index=False)
    print("Análise: PASS. Matriz histórica, 6 remapeamentos, 8 recortes e 8 desenhos/tamanhos.",flush=True)
    print(pd.DataFrame(domain_rows)[["case_id","accepted_N","thermo_fraction_pct","mismatch_pct","thermo_error_pct","diffusive_error_pct"]].to_string(index=False))
    print(pd.DataFrame(sampling_rows)[["case_id","candidate_N","accepted_N","mismatch_pct","thermo_as_diffusive_N","diffusive_as_thermo_N"]].to_string(index=False))


if __name__=="__main__":main()
