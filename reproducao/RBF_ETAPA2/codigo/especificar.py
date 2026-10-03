"""Materializa a ordem e as caixas dos 17 parâmetros a partir das fontes."""
from pathlib import Path
import csv,json,yaml
ROOT=Path(__file__).resolve().parents[1];model=ROOT/'original_reproduzivel'
policy=yaml.safe_load((model/'config/r1d2b_production_coefficient_policy.yaml').read_text())
with (model/'data/literature/literature_bounds.csv').open(newline='') as f:
 bounds={x['bound_id']:x for x in csv.DictReader(f)}
spec=[
 ('a_center','a_c','g','R1D-A-COEFF'),('a_inner','a_i','g','R1D-A-COEFF'),('a_outer','a_o','g','R1D-A-COEFF'),
 ('mu_g_inner_nm','mu_G_i','g','R1D-MU-G-INNER'),('mu_g_outer_nm','mu_G_o','g','R1D-MU-G-OUTER'),
 ('sigma_g_center_nm','sigma_G_c','g','R1D-SIGMA-G-CENTER'),('sigma_g_inner_nm','sigma_G_i','g','R1D-SIGMA-G-INNER'),
 ('sigma_g_outer_nm','sigma_G_o','g','R1D-SIGMA-G-OUTER'),
 ('b_center','b_c','log_d','R1D-B-COEFF'),('b_inner','b_i','log_d','R1D-B-COEFF'),('b_outer','b_o','log_d','R1D-B-COEFF'),
 ('mu_d_inner_nm','mu_D_i','log_d','R1D-MU-D-INNER'),('mu_d_outer_nm','mu_D_o','log_d','R1D-MU-D-OUTER'),
 ('sigma_d_center_nm','sigma_D_c','log_d','R1D-SIGMA-D'),('sigma_d_inner_nm','sigma_D_i','log_d','R1D-SIGMA-D'),
 ('sigma_d_outer_nm','sigma_D_o','log_d','R1D-SIGMA-D'),('D_bulk_nm2_ns','D_bulk','D','R1D-DBULK')
]
rows=[]
for i,(name,symbol,field,boundid) in enumerate(spec):
 q=bounds[boundid]
 if name.startswith('a_'):lo,hi=policy['coefficient_bounds']['a_j'];authority='config/r1d2b_production_coefficient_policy.yaml'
 elif name.startswith('b_'):lo,hi=policy['coefficient_bounds']['b_j'];authority='config/r1d2b_production_coefficient_policy.yaml'
 else:lo,hi=float(q['frozen_low']),float(q['frozen_high']);authority='data/literature/literature_bounds.csv'
 rows.append({'sobol_column_base0':i,'csv_field':name,'symbol':symbol,'profile':field,
              'unit':q['unit'],'low':lo,'high':hi,'transform':'low+(high-low)*u_j',
              'bound_id':boundid,'numeric_authority':authority,
              'evidence_basis':q['evidence_basis'],'notes':q['notes']})
assert len(rows)==17
with (ROOT/'PARAMETROS_17.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'evidencias/especificacao_parametros.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print('17 parâmetros transcritos, com ordem, unidade, transformação e fonte numérica.')
