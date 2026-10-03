"""Confere arquivos e resultados encerrados sem reexecutar campanhas científicas."""
from pathlib import Path
import argparse, hashlib, json, re, unicodedata
import numpy as np
import pandas as pd

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def frame(p): return pd.read_csv(p,float_precision='round_trip')
def norm(s): return re.sub(r'\s+','',unicodedata.normalize('NFKC',s))

def audit(b):
 r=b/'reproducao';s2=r/'RBF_ETAPA2';s3=r/'RBF_ETAPA3';s4=r/'RBF_ETAPA4';s5=r/'RBF_ETAPA5'
 m=json.loads((b/'MANIFESTO_SHA256.json').read_text())
 assert m and all((b/k).is_file() and sha(b/k)==v for k,v in m.items())
 q={'status':'PASS_DOCUMENTAL_E_NUMERICO','scientific_campaign_reexecuted':False,'base_member_hashes_verified':len(m)}
 params=frame(s2/'PARAMETROS_17.csv');fields=params.csv_field.tolist()
 assert len(fields)==17 and params.sobol_column_base0.tolist()==list(range(17))
 old=frame(s5/'originais/producao_historica.csv.gz');new=frame(s5/'dados/producao_revisao_numerica.csv.gz')
 assert sha(s5/'originais/producao_historica.csv.gz')=='17669c5c0bd763181c0198a383e4f5c8cc643138ec7c0c4397c9f0cda6ba1cdb'
 assert len(old)==len(new)==56926
 assert np.array_equal(old[['sobol_index']+fields],new[['sobol_index']+fields])
 changed=np.any(old.to_numpy()!=new[old.columns].to_numpy(),axis=1)
 ids=old.sobol_index[changed].tolist();assert ids==[35952,39910,41020,41752,50510,57164]
 om=old.g_max+np.log(old.d_min)
 assert np.array_equal(np.sign(old.deltaC),np.sign(new.deltaC))
 assert np.array_equal(np.sign(om),np.sign(new.proxy_margin))
 t=new.deltaC>0;pt=new.proxy_margin>0
 matrix=[[int((t&pt).sum()),int((t&~pt).sum())],[int((~t&pt).sum()),int((~t&~pt).sum())]]
 assert matrix==[[29161,7],[4366,23392]]
 assert not (new.deltaC==0).any() and not (new.proxy_margin==0).any()
 residual=float(np.max(np.abs(new.deltaC-new.proxy_margin-new.rho)));assert residual<3e-15
 cf=frame(s2/'dados/candidatos_e_filtros.csv.gz');assert len(cf)==65536 and cf.accepted.sum()==56926
 p=['g_lower','g_upper','d_lower','d_upper']
 ex=[int((cf.first_failure==v).sum()) for v in p];nx=[int((~cf[v+'_ok']).sum()) for v in p]
 assert ex==[893,294,6920,503] and nx==[893,294,7051,557]
 q['production']={'N':len(new),'candidate_N':len(cf),'rejected_N':sum(ex),'matrix':matrix,
   'matrix_orientation':'linhas integrada T/D; colunas proxy T/D','agreement_pct':100*52553/56926,
   'global_error_pct':100*4373/56926,'D_error_pct':100*4366/27758,'T_error_pct':100*7/29168,
   'revised_ids':ids,'all_17_parameters_and_labels_preserved':True,'identity_max_abs_error':residual,
   'filter_exclusive':dict(zip(p,ex)),'filter_nonexclusive':dict(zip(p,nx))}
 neg=old.AG<0;mis=(old.deltaC<0)&(om>0);assert neg.sum()==23952 and (neg&mis).sum()==1174
 q['asymmetry']={'AG_negative_N':int(neg.sum()),'D_to_T_AG_negative_N':int((neg&mis).sum()),'D_to_T_AG_nonnegative_N':int((~neg&mis).sum())}
 alpha={str(a):int((np.sign(old.deltaC+(2*a-1)*old.interaction)!=np.sign(old.deltaC)).sum()) for a in [0,.25,.5,.75,1]}
 assert alpha=={'0':1508,'0.25':775,'0.5':0,'0.75':663,'1':1243}
 robust=int((abs(old.deltaC)>abs(old.interaction)+1e-12).sum());assert robust==54175
 q['attribution']={'changed_vs_Shapley':alpha,'stable_for_all_alpha_N':robust}
 pair=frame(s3/'dados/resistencias_contribuicoes.csv').set_index('profile_id').loc[['original_A','original_B']]
 assert pair.max_g_analytic.eq(1).all() and pair.d_min_analytic.eq(np.exp(-1)).all()
 assert pair.deltaC.iloc[0]>0 and pair.deltaC.iloc[1]<0
 q['constructive_pair']={'profile_ids':pair.index.tolist(),'deltaC':pair.deltaC.tolist(),
   'analytic_proof_location':'reproducao/RBF_ETAPA3/PROVA_NAO_IDENTIFICABILIDADE.md','proof_newly_executed':False}
 dom=frame(s4/'dados/resumo_dominio.csv');remap=dom[dom.case_id.isin(['amplitude_G_080','amplitude_D_080','larguras_G_centro50','larguras_D_metade_inferior','larguras_D_metade_superior','posicoes_centro50'])]
 assert len(remap)==6 and remap.accepted_N.tolist()==[57923,63074,57246,60932,54747,57982]
 sam=frame(s4/'dados/resumo_amostragem.csv');scr=sam[sam.case_id.str.startswith('sobol_embaralhado')]
 ext=sam[sam.case_id=='prefixo_sobol_17'].iloc[0]
 assert len(scr)==4 and ext.accepted_N==113905 and ext.mismatch_N==8822
 q['sensitivities']={'remapping':remap[['case_id','accepted_N','mismatch_pct','diffusive_error_pct']].to_dict('records'),
   'scramble_global_range_pct':[float(scr.mismatch_pct.min()),float(scr.mismatch_pct.max())],
   'extension_accepted_N':int(ext.accepted_N),'extension_mismatch_N':int(ext.mismatch_N)}
 prec=frame(s5/'dados/precisao_todos_originais.csv.gz');assert len(prec)==56926 and prec.boundary_0p10_original.sum()==985
 for c in ['interval_reaches_zero','proxy_interval_reaches_zero','integrated_label_changed','proxy_label_changed']:assert not prec[c].any(),c
 grids={str(k):int(v) for k,v in prec.assessment_grid_N.value_counts().sort_index().items()}
 assert grids=={'6401':50411,'25601':6119,'51201':396}
 assert prec.deltaC_error_estimate.max()<1e-7 and prec.P_relative_error_estimate.max()<1e-7
 quad=frame(s5/'dados/quadpack_comparacao.csv');assert len(quad)==96
 fp=frame(s5/'dados/fokker_planck_direto.csv');assert len(fp)==740 and not fp.integrated_sign_differs_from_ISD.any()
 fperr={str(k):float(v) for k,v in fp.groupby('grid_N').relative_FP_vs_ISD.max().items()}
 assert fperr['3201']>5e-5 and all(fperr[str(k)]<=5e-5 for k in [6401,12801,25601])
 q['numerical']={'near_boundary_N':int(prec.boundary_0p10_original.sum()),'assessment_grids':grids,'changed_labels_N':0,
   'estimated_intervals_reaching_zero_N':0,'max_estimated_delta_error':float(prec.deltaC_error_estimate.max()),
   'max_estimated_relative_P_error':float(prec.P_relative_error_estimate.max()),'quad_profile_N':len(quad),
   'max_quad_relative_P_difference':float(quad.relative_uniform25601_vs_quad_P.max()),'FP_profile_N':185,
   'FP_systems':len(fp)*4,'FP_max_relative_P_by_grid':fperr,
   'FP_max_flux_nonuniformity':float(fp.max_flux_nonuniformity.max()),
   'FP_max_scaled_residual':float(fp.max_scaled_equation_residual.max()),'certified_errors':False}
 bd=frame(s5/'dados/brownian_novo_resumo.csv');bm=frame(s5/'dados/brownian_margem_fronteira.csv')
 assert len(bd)==12 and bd.total_paths.sum()==1966080 and not bd.unresolved_total.any()
 assert bd.status.eq('PASS').all() and len(bm)==3 and bm.zero_inside_MC95.all()
 inside=(bd.P_FP_nm_ns>=bd.P_BD_MC95_lower)&(bd.P_FP_nm_ns<=bd.P_BD_MC95_upper);assert inside.sum()==11
 q['brownian']={'runs':len(bd),'trajectories':int(bd.total_paths.sum()),'censored_N':0,'P_MC95_including_FP_N':int(inside.sum()),
   'exception':bd.loc[~inside,['case_id','method','dt_ns','zscore_BD_vs_FP']].to_dict('records'),
   'maximum_relative_P_difference':float(bd.relative_BD_vs_FP.max()),
   'boundary_intervals':bm[['method','dt_ns','deltaC_MC95_lower','deltaC_MC95_upper']].to_dict('records'),
   'new_control_class_resolved':False,'historical_P4_limit_retained':True}
 q['scientific_limits']=['Estimated numerical errors, not certified bounds.','Physical uncertainty not quantified.','BD covers selected controls, not the full ensemble; boundary control remains unresolved.','Historical P4 is censored with incomplete denominators.','FP and BD verify the same formalism; no experimental validation.']
 return q

if __name__=='__main__':
 ap=argparse.ArgumentParser(description='Verify frozen public files and reported results; no scientific campaign is executed.')
 ap.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1])
 ap.add_argument('--out',type=Path,help='Optional external JSON report; keep outside the frozen package.')
 args=ap.parse_args();q=audit(args.base.resolve())
 if args.out:
  args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(q,ensure_ascii=False,indent=2))
