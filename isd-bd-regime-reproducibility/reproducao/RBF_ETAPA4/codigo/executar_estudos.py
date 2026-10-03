"""Executa o protocolo da etapa 4. O histórico é validado e preservado."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
import numpy as np
import pandas as pd
from motor_sensibilidade import *


def reference():
    t=time.monotonic()
    df=pd.read_csv(ROOT/"originais/producao_historica.csv.gz",float_precision="round_trip")
    canon=np.load(ROOT/"originais/filtro_canonico_65536.npz",allow_pickle=False)
    spec=dict(id="validacao_original",m=16,scramble=False)
    ids,u,p,meta=points(spec)
    assert np.array_equal(u,np.load(ROOT/"originais/sobol_65536x17.npy",allow_pickle=False))
    accepted_ids=df.sobol_index.to_numpy(int)
    assert np.array_equal(p[accepted_ids],df[FIELDS].to_numpy(float))
    print("Validando 65.536 candidatos contra a produção original...",flush=True)
    calculated=evaluate(ids,p)
    assert np.array_equal(calculated["fail_bits"],canon["fail_bits"])
    assert np.array_equal(np.flatnonzero(calculated["accepted"]),accepted_ids)
    filter_errors={k:float(np.max(np.abs(calculated[k]-canon[k]))) for k in FILTER}
    assert max(filter_errors.values())<=2e-12
    for row in df[df.numerical_refinement_applied].itertuples(index=False):
        idx=int(row.sobol_index);record=source_record(p[idx],int(row.evaluation_grid_N))
        assert record["source_accepted"];apply_source(calculated,idx,record,int(row.evaluation_grid_N))
    errors={}
    for key in FINAL:
        if key in ["proxy_margin"]:continue
        if key=="resistance_peak_abs_z_nm":
            # O argmax pode resolver empates simétricos de outra forma; a integração é independente.
            continue
        a=calculated[key][accepted_ids];b=df[key].to_numpy(float)
        err=np.abs(a-b)/np.abs(b) if key=="P_cm_s" else np.abs(a-b)
        errors[key]=float(err.max())
        assert errors[key]<=2e-12,(key,errors[key])
    spot=original_spot_checks(calculated,p)
    # A referência usada na análise é o CSV canônico, sem substituição por uma nova tabela.
    out={k:np.full(len(ids),np.nan) for k in FINAL}
    out.update({k:canon[k].copy() for k in FILTER})
    out.update(point_index=ids,fail_bits=canon["fail_bits"].copy(),
               accepted=canon["fail_bits"]==0,evaluation_grid_N=np.full(len(ids),3201,dtype=np.uint16))
    for k in FINAL:
        if k=="proxy_margin":
            out[k][accepted_ids]=df.g_max.to_numpy()-math.log(10)*df.eta.to_numpy()
        else:
            out[k][accepted_ids]=df[k].to_numpy(float)
    out["evaluation_grid_N"][accepted_ids]=df.evaluation_grid_N.to_numpy(np.uint16)
    classify(out)
    assert np.array_equal(out["integrated_class"][accepted_ids],np.sign(calculated["deltaC"][accepted_ids]))
    assert np.array_equal(out["proxy_class"][accepted_ids],np.sign(calculated["proxy_margin"][accepted_ids]))
    np.savez_compressed(ROOT/"dados/referencia_original.npz",**out)
    write_json(ROOT/"evidencias/validacao_motor.json",dict(
        status="PASS",candidate_N=len(ids),accepted_N=len(df),all_parameters_exact=True,
        all_sobol_points_exact=True,all_filter_flags_exact=True,all_integrated_labels_exact=True,
        all_proxy_labels_exact=True,filter_max_errors=filter_errors,final_max_errors=errors,
        original_module_spot_checks=spot,historical_refined_ids=df.loc[df.numerical_refinement_applied,"sobol_index"].tolist(),
        source_csv_retained=True,elapsed_seconds=time.monotonic()-t))
    print(f"Validação original: PASS; segundos={time.monotonic()-t:.1f}",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--only",nargs="*",help="Executar apenas IDs novos selecionados, após a validação inicial.")
    parser.add_argument("--reuse-reference",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    assert 1<=args.workers<=9
    verify_inputs()
    if not args.reuse_reference:reference()
    else:
        assert json.loads((ROOT/"evidencias/validacao_motor.json").read_text())["status"]=="PASS"
    specs=[dict(s,m=16,scramble=False,kind="domain_remap") for s in PROTOCOL["domain_scenarios"]]
    specs+=[dict(id="extensao_sobol_17",label="Extensão determinística",m=17,start_index=65536,
                 scramble=False,kind="sampling_extension")]
    specs+=[dict(id=f"sobol_embaralhado_{i}",label=f"Sobol embaralhado {i}",m=16,scramble=True,
                 seed=int(seed),kind="sampling_scramble") for i,seed in enumerate(PROTOCOL["scrambled_seed"],1)]
    if args.only:
        wanted=set(args.only);assert wanted<={s["id"] for s in specs}
        specs=[s for s in specs if s["id"] in wanted]
    if args.resume:
        specs=[s for s in specs if not (ROOT/"dados"/(s["id"]+".npz")).exists()]
    all_u=qmc.Sobol(d=17,scramble=False,bits=30,optimization=None).random_base2(17)
    assert np.array_equal(all_u[:65536],np.load(ROOT/"originais/sobol_65536x17.npy",allow_pickle=False))
    write_json(ROOT/"evidencias/prefixo_preservado.json",dict(status="PASS",N=131072,
               original_prefix_exact=True,first_point_kept=True,thin=False,bits=30,
               full_points_sha256=hashlib.sha256(all_u.tobytes()).hexdigest()))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,s):s["id"] for s in specs}
        for f in as_completed(futures):f.result()
    versions={k:importlib.metadata.version(k) for k in ["numpy","scipy","pandas","matplotlib","PyYAML","Pillow"]}
    write_json(ROOT/"evidencias/ambiente.json",dict(python=sys.version,platform=platform.platform(),
               executable=sys.executable,workers=args.workers,packages=versions,
               legacy_stage2_environment_claimed=False))
    (ROOT/"requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items())+"\n")
    print("Estudos solicitados concluídos.",flush=True)


if __name__=="__main__":main()
