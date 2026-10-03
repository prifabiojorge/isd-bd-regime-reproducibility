"""Executa o materializador original; instrumenta apenas o registro dos filtros.

Uso: python codigo/reproduzir.py
O código científico e o materializador permanecem byte a byte preservados.
"""
from pathlib import Path
import csv, gzip, hashlib, json, os, platform, runpy, sys, time
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'original_reproduzivel'
DATA = ROOT / 'dados'
DATA.mkdir(exist_ok=True)
os.environ['ISDBD_ROOT'] = str(MODEL)
os.environ['ISDBD_JOB'] = str(DATA)
sys.path.insert(0, str(MODEL / 'src'))
import isdbd_regime.profiles as profiles

PARAMS = ['a_center', 'a_inner', 'a_outer', 'mu_g_inner_nm', 'mu_g_outer_nm',
          'sigma_g_center_nm', 'sigma_g_inner_nm', 'sigma_g_outer_nm',
          'b_center', 'b_inner', 'b_outer', 'mu_d_inner_nm', 'mu_d_outer_nm',
          'sigma_d_center_nm', 'sigma_d_inner_nm', 'sigma_d_outer_nm', 'D_bulk_nm2_ns']
FIELDS = ['sobol_index'] + PARAMS + ['accepted', 'geometry_ok', 'g_ok', 'd_ok',
         'g_lower_ok', 'g_upper_ok', 'd_lower_ok', 'd_upper_ok',
         'first_failure', 'all_failures', 'filter_grid_N', 'filter_atol',
         'filter_g_min', 'filter_g_max', 'filter_d_min', 'filter_d_max',
         'observed_g_min', 'observed_g_max', 'observed_d_min', 'observed_d_max']
original_build = profiles.build_profile
original_accept = profiles.evaluate_profile_acceptance
last_build = None
candidate_calls = 0
refined_checks = []
fail_counts = dict.fromkeys(['g_lower', 'g_upper', 'd_lower', 'd_upper'], 0)
exclusive = dict.fromkeys(fail_counts, 0)
combinations = {}
start = time.time()
versions = {n: importlib.metadata.version(n) for n in
            ['numpy','scipy','pandas','matplotlib','h5py','PyYAML','numba','pytest','pillow']}
(ROOT / 'evidencias/ambiente_execucao.json').write_text(json.dumps({
    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
    'machine': platform.machine(), 'packages': versions,
    'historical_platform_recreated': False,
    'source_hash': hashlib.sha256((ROOT/'codigo/materialize_original.py').read_bytes()).hexdigest()
}, indent=2)+'\n')

def traced_build(*args, **kwargs):
    global last_build
    result = original_build(*args, **kwargs)
    last_build = (result, args)
    return result

def traced_accept(profile, g_min, g_max, d_min, d_max, atol=2e-12):
    global candidate_calls
    result = original_accept(profile, g_min, g_max, d_min, d_max, atol)
    if candidate_calls >= 65536:
        refined_checks.append({'grid_N': len(profile.z_nm), **result})
        return result
    assert last_build is not None and last_build[0] is profile
    z, gg, dg, a, b, db = last_build[1]
    values = [a.center,a.inner,a.outer,gg.mu_inner_nm,gg.mu_outer_nm,
              gg.sigma_center_nm,gg.sigma_inner_nm,gg.sigma_outer_nm,
              b.center,b.inner,b.outer,dg.mu_inner_nm,dg.mu_outer_nm,
              dg.sigma_center_nm,dg.sigma_inner_nm,dg.sigma_outer_nm,db]
    oks = [result['g_min_observed'] >= g_min-atol,
           result['g_max_observed'] <= g_max+atol,
           result['d_min_observed'] >= d_min-atol,
           result['d_max_observed'] <= d_max+atol]
    failures = [k for k,ok in zip(fail_counts,oks) if not ok]
    assert result['accepted'] == all(oks)
    for k in failures: fail_counts[k] += 1
    if failures:
        exclusive[failures[0]] += 1
        key = '+'.join(failures)
        combinations[key] = combinations.get(key,0)+1
    row = dict(zip(PARAMS,values))
    row.update(sobol_index=candidate_calls, accepted=result['accepted'], geometry_ok=True,
               g_ok=result['g_ok'], d_ok=result['d_ok'],
               g_lower_ok=oks[0], g_upper_ok=oks[1], d_lower_ok=oks[2], d_upper_ok=oks[3],
               first_failure=failures[0] if failures else '',all_failures='+'.join(failures),
               filter_grid_N=len(z),filter_atol=atol,
               filter_g_min=g_min,filter_g_max=g_max,filter_d_min=d_min,filter_d_max=d_max,
               observed_g_min=result['g_min_observed'],observed_g_max=result['g_max_observed'],
               observed_d_min=result['d_min_observed'],observed_d_max=result['d_max_observed'])
    writer.writerow(row)
    candidate_calls += 1
    if candidate_calls % 8192 == 0:
        print(f'candidatos_registrados={candidate_calls}; segundos={time.time()-start:.1f}',flush=True)
    return result

profiles.build_profile = traced_build
profiles.evaluate_profile_acceptance = traced_accept
with (DATA/'candidatos_e_filtros.csv.gz').open('wb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
        import io
        with io.TextIOWrapper(gz,encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n')
            writer.writeheader()
            runpy.run_path(str(ROOT/'codigo/materialize_original.py'),run_name='__main__')
assert candidate_calls == 65536
(ROOT/'evidencias/filtros_reproduzidos.json').write_text(json.dumps({
    'candidate_calls':candidate_calls,'priority':list(fail_counts),
    'nonexclusive_failure_counts':fail_counts,'exclusive_first_failure_counts':exclusive,
    'overlap_combinations':combinations,'refined_filter_checks':refined_checks,
    'atol':2e-12,'grid_N':3201,
    'sampling_claim':'Filter evaluated at grid points; no certification between nodes.'
},indent=2)+'\n')
print('Registro adicional dos filtros concluído; algoritmo original preservado.',flush=True)
