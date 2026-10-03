"""Seleção determinística de cobertura a partir dos resultados anteriores."""
import argparse
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import qmc
from numerica import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--etapa4",type=Path,default=ROOT.parent/"RBF_ETAPA4")
    args=parser.parse_args()
    if len(list((args.etapa4/"dados").glob("*.npz")))<11:
        raise FileNotFoundError("A recomposição da seleção exige o pacote da etapa 4 via --etapa4. "
                                "Os cálculos da etapa 5 usam diretamente dados/casos_selecionados.csv.gz, "
                                "incluído e protegido por SHA-256 neste pacote.")
    provenance_path=ROOT/"evidencias/proveniencia_etapa4.json"
    if provenance_path.exists():
        for prior in json.loads(provenance_path.read_text())["additional_cases"]:
            source=args.etapa4/"dados"/(prior["case_id"]+".npz")
            assert source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==prior["source_npz_sha256"],source
    verify_inputs()
    base=pd.read_csv(ROOT/"originais/producao_historica.csv.gz",float_precision="round_trip")
    p=base[FIELDS].to_numpy(float)
    grad={k:np.empty(len(base)) for k in ["max_abs_gprime","max_abs_ellprime","max_abs_ito_drift"]}
    z=np.linspace(-H,H,PROTOCOL["integration"]["gradient_rank_grid"])
    for start in range(0,len(base),64):
        end=min(start+64,len(base));theta=p[start:end]
        g,ell,gp,ep=fields_batch(theta,z,derivatives=True)
        D=theta[:,16,None]*np.exp(ell)
        for key,value in zip(grad,[gp,ep,D*(ep-gp)]):
            grad[key][start:end]=np.max(np.abs(value),axis=1)
    gradients=pd.DataFrame(dict(sobol_index=base.sobol_index.to_numpy(int),**grad))
    gradients.to_csv(ROOT/"dados/gradientes_original.csv",index=False,float_format="%.17g")
    # Conferência analítica com o módulo recuperado, incluindo os extremos.
    checks=np.unique(np.r_[np.linspace(0,len(base)-1,64,dtype=int),
                            *[np.argsort(v)[-8:] for v in grad.values()]])
    maxerrors=dict(gprime=0.,ellprime=0.,ito_drift=0.)
    for i in checks:
        _,q=source_profile(p[i],len(z))
        g,ell,gp,ep=fields_batch(p[i],z,True)
        values=[gp[0],ep[0],p[i,16]*np.exp(ell[0])*(ep[0]-gp[0])]
        for key,a,b in zip(maxerrors,values,[q.g_prime_per_nm,q.ell_prime_per_nm,q.ito_drift_nm_ns]):
            maxerrors[key]=max(maxerrors[key],float(np.max(np.abs(a-b))))
    assert max(maxerrors.values())<=1e-11,maxerrors
    write_json(ROOT/"evidencias/validacao_gradientes.json",dict(status="PASS",N=len(checks),
               sampled_grid_N=len(z),maximum_errors=maxerrors,continuous_maximum_certified=False))
    reasons=defaultdict(set)
    def mark(ids,reason):
        for i in np.asarray(ids,int):reasons[int(i)].add(reason)
    idx=base.sobol_index.to_numpy(int)
    mark(base.loc[np.abs(base.deltaC)<=.10,"sobol_index"],"fronteira_0p10")
    m=base.g_max.to_numpy()-math.log(10)*base.eta.to_numpy()
    mark(idx[np.sign(base.deltaC.to_numpy())!=np.sign(m)],"discordancia_proxy")
    hist=pd.read_csv(ROOT/"originais/r2d2bfix2_isd_refinement_audit.csv")
    mark(hist.sobol_index,"auditoria_historica_1024")
    original_fp=pd.read_csv(ROOT/"originais/r2d2bfix2_fp_crosscheck_subset.csv").sobol_index.to_numpy(int)
    mark(original_fp,"fp_historico_128")
    for key,value in grad.items():mark(idx[np.argsort(value,kind="stable")[-64:]],key)
    for column,ascending in [("d_min",True),("P_cm_s",True),("P_cm_s",False)]:
        mark(base.sort_values([column,"sobol_index"],ascending=[ascending,True]).head(64).sobol_index,
             "extremo_"+column+("_inferior" if ascending else "_superior"))
    fp_ids=set(original_fp)
    for value in grad.values():fp_ids.update(idx[np.argsort(value,kind="stable")[-8:]])
    for sign in [-1,1]:
        rows=base[np.sign(base.deltaC)==sign].assign(abs_delta=lambda x:np.abs(x.deltaC))
        fp_ids.update(rows.sort_values(["abs_delta","sobol_index"]).head(16).sobol_index)
    inverse=idx[(np.sign(base.deltaC.to_numpy())==1)&(np.sign(m)==-1)]
    fp_ids.update(inverse)
    mark(list(fp_ids),"fp_adicional")
    allrows=[]
    for row in base.itertuples(index=False):
        i=int(row.sobol_index)
        if i not in reasons:continue
        r={key:getattr(row,key) for key in FIELDS+METRICS if hasattr(row,key)}
        r.update(source_case_id="referencia_original",point_index=i,
                 source_grid_N=int(row.evaluation_grid_N),
                 proxy_margin=float(row.g_max-math.log(10)*row.eta),
                 mesh_reasons=";".join(sorted(reasons[i])),fp_selected=i in fp_ids,quad_selected=False)
        allrows.append(r)
    provenance=[]
    for casepath in sorted((args.etapa4/"dados").glob("*.npz")):
        if casepath.stem in {"referencia_original","recortes_supressao"}:continue
        meta=json.loads((args.etapa4/"evidencias"/(casepath.stem+".json")).read_text())
        data=np.load(casepath,allow_pickle=False)
        valid=np.flatnonzero(data["accepted"])
        selections=defaultdict(set)
        for field,n,absolute,reverse in [("deltaC",32,True,False),("proxy_margin",8,True,False),
                                          ("d_min",16,False,False),("g_max",8,False,True)]:
            value=data[field][valid];value=np.abs(value) if absolute else value
            order=np.argsort(-value if reverse else value,kind="stable")[:n]
            for i in valid[order]:selections[int(i)].add("adicional_"+field)
        kwargs=dict(d=17,scramble=meta["scramble"],bits=30,optimization=None)
        if meta["scramble"]:kwargs["rng"]=int(meta["seed"])
        u=qmc.Sobol(**kwargs).random_base2(int(meta["m"]))
        lo,hi=np.asarray(meta["lower_bounds"]),np.asarray(meta["upper_bounds"])
        theta=lo+(hi-lo)*u
        start_index=int(meta.get("start_index",0))
        assert hashlib.sha256(u[start_index:].tobytes()).hexdigest()==meta["input_points_sha256"]
        # A extensão registra somente a segunda metade; point_index continua global.
        assert hashlib.sha256(theta[start_index:].tobytes()).hexdigest()==meta["input_parameter_sha256"]
        nearest=valid[np.argsort(np.abs(data["deltaC"][valid]),kind="stable")[:2]]
        fp_new=set(map(int,nearest))
        for i in sorted(selections):
            point=int(data["point_index"][i]);r={field:float(theta[point,j]) for j,field in enumerate(FIELDS)}
            r.update({key:float(data[key][i]) for key in METRICS})
            r.update(source_case_id=casepath.stem,point_index=point,source_grid_N=int(data["evaluation_grid_N"][i]),
                     mesh_reasons=";".join(sorted(selections[i])),fp_selected=i in fp_new,quad_selected=False)
            allrows.append(r)
        provenance.append(dict(case_id=casepath.stem,source_npz_sha256=hashlib.sha256(casepath.read_bytes()).hexdigest(),
                               source_points_sha256=meta["input_points_sha256"],source_parameters_sha256=meta["input_parameter_sha256"],
                               selected_N=len(selections),parent_package_library_file_id="libfile_531398dec0d88191b70bdb35013dc175"))
    chosen=pd.DataFrame(allrows).sort_values(["source_case_id","point_index"]).reset_index(drop=True)
    assert not chosen.duplicated(["source_case_id","point_index"]).any()
    # Prioridades definidas por margens, erros inversos e gradientes, sem resultado novo de integração.
    quad_keys=[]
    def extend(keys):
        for key in keys:
            if key not in quad_keys:quad_keys.append(key)
    extend([("referencia_original",int(i)) for i in inverse])
    extend([("referencia_original",36312)])
    for case,group in chosen.groupby("source_case_id",sort=True):
        top=group.assign(abs_delta=lambda x:np.abs(x.deltaC)).sort_values(["abs_delta","point_index"]).head(4)
        extend(list(zip(top.source_case_id,top.point_index.astype(int))))
    for value in grad.values():extend([("referencia_original",int(i)) for i in idx[np.argsort(value)[-8:]]])
    top=chosen.assign(abs_delta=lambda x:np.abs(x.deltaC)).sort_values(["abs_delta","source_case_id","point_index"])
    extend(list(zip(top.source_case_id,top.point_index.astype(int))))
    quad_keys=set(quad_keys[:96])
    chosen["quad_selected"]=[(r.source_case_id,int(r.point_index)) in quad_keys for r in chosen.itertuples()]
    path=ROOT/"dados/casos_selecionados.csv.gz"
    chosen.to_csv(path,index=False,float_format="%.17g",compression=dict(method="gzip",mtime=0))
    write_json(ROOT/"evidencias/proveniencia_etapa4.json",dict(status="PASS",additional_cases=provenance,
               original_csv_sha256=hashlib.sha256((ROOT/"originais/producao_historica.csv.gz").read_bytes()).hexdigest(),
               includes_complete_stage4_ensembles=False,selected_inputs_self_contained=True))
    manifest=json.loads((ROOT/"evidencias/entradas_sha256.json").read_text())
    manifest[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(ROOT/"evidencias/entradas_sha256.json",manifest)
    write_json(ROOT/"evidencias/selecao.json",dict(status="PASS",original_accepted_N=len(base),
               selected_total_N=len(chosen),selected_original_N=int((chosen.source_case_id=="referencia_original").sum()),
               selected_additional_N=int((chosen.source_case_id!="referencia_original").sum()),
               original_boundary_all_N=int((np.abs(base.deltaC)<=.1).sum()),original_proxy_mismatch_all_N=4373,
               fp_N=int(chosen.fp_selected.sum()),quad_N=int(chosen.quad_selected.sum()),
               counts_by_case=chosen.groupby("source_case_id").size().to_dict()))
    print(json.dumps(json.loads((ROOT/"evidencias/selecao.json").read_text()),ensure_ascii=False),flush=True)


if __name__=="__main__":main()
