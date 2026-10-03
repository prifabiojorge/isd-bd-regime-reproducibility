"""Completa todas as colunas dos seis registros que requerem revisão.

Usa os módulos escalares históricos sem modificá-los. A tabela histórica
continua byte a byte intacta; esta função só edita a nova vista derivada.
"""
from numerica import *


def canonical_values(theta, n):
    z, p = source_profile(theta, int(n))
    zeros = np.zeros(len(z))
    flat = np.full(len(z), float(theta[16]))
    states = [isd_log_trapezoid(z, zeros, flat),
              isd_log_trapezoid(z, p.g, flat),
              isd_log_trapezoid(z, zeros, p.D_nm2_ns),
              isd_log_trapezoid(z, p.g, p.D_nm2_ns)]
    r0, rg, rd, rgd = [s.log_resistance for s in states]
    ag, ad = rg-r0, rd-r0
    interaction = rgd-rg-rd+r0
    gmax, dmin = float(p.g.max()), float(p.d_ratio.min())
    eta = -math.log10(dmin)
    wg, wd = math.exp(ag-gmax), math.exp(ad-math.log(10)*eta)
    return dict(g_min=float(p.g.min()), g_max=gmax, d_min=dmin,
                d_max=float(p.d_ratio.max()), eta=eta,
                P_nm_ns=states[3].permeability_nm_per_ns,
                P_cm_s=states[3].permeability_cm_per_s,
                log10_P_cm_s=math.log10(states[3].permeability_cm_per_s),
                logR0=r0, logRG=rg, logRD=rd, logRGD=rgd,
                AG=ag, AD=ad, interaction=interaction,
                CG=ag+interaction/2, CD=ad+interaction/2,
                deltaC=ag-ad, wG=wg, wD=wd, rho=math.log(wg/wd),
                proxy_margin=gmax-math.log(10)*eta,
                resistance_peak_abs_z_nm=float(abs(z[np.argmax(p.g-p.ell)])))


def complete_revision(revised, theta, grids, requires):
    rows=[]
    for j in np.flatnonzero(requires):
        values=canonical_values(theta[j], grids[j])
        checks=source_metrics(theta[j], int(grids[j]))
        assert max(abs(values[k]/checks[k]-1) if k=="P_cm_s" else
                   abs(values[k]-checks[k]) for k in METRICS)<2e-12
        for key, value in values.items():revised.loc[j,key]=value
        revised.loc[j,"evaluation_grid_N"]=int(grids[j])
        revised.loc[j,"numerical_refinement_applied"]=True
        rows.append(dict(sobol_index=int(revised.sobol_index.iloc[j]),
                         updated_fields=list(values),grid_N=int(grids[j]),
                         consistent_with_historical_scalar_modules=True))
    return rows


def main():
    verify_inputs()
    base=pd.read_csv(ROOT/"originais/producao_historica.csv.gz",float_precision="round_trip")
    revised=pd.read_csv(ROOT/"dados/producao_revisao_numerica.csv.gz",float_precision="round_trip")
    required=revised.stage5_revision_applied.to_numpy(bool)
    grids=revised.stage5_assessment_grid_N.to_numpy(int)
    details=complete_revision(revised,base[FIELDS].to_numpy(float),grids,required)
    revised.to_csv(ROOT/"dados/producao_revisao_numerica.csv.gz",index=False,
                   float_format="%.17g",compression=dict(method="gzip",mtime=0))
    # Não permitir alterações em parâmetros nem em valores de linhas não revistas.
    reread=pd.read_csv(ROOT/"dados/producao_revisao_numerica.csv.gz",float_precision="round_trip")
    assert np.array_equal(reread[FIELDS].to_numpy(),base[FIELDS].to_numpy())
    assert np.array_equal(reread.loc[~required,base.columns].to_numpy(),
                          base.loc[~required].to_numpy())
    write_json(ROOT/"evidencias/revisao_producao_integridade.json",dict(status="PASS",N=int(required.sum()),
        immutable_history_preserved=True,all_parameters_unchanged=True,
        unaffected_records_numerically_identical=True,all_dependent_columns_updated=True,rows=details))
    print("Revisão completa e consistente:",int(required.sum()),"registros.")


if __name__=="__main__":main()
