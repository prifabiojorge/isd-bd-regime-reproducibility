"""Verifica os contraexemplos da etapa 3, sem alterar o ensemble histórico.

Uso: python codigo/executar_etapa3.py [--historico CSV_ORIGINAL.gz]
O histórico completo é necessário apenas para a primeira extração da projeção.
No pacote entregue, dados/projecao_historica.csv.gz permite repetir a métrica.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.integrate import cumulative_trapezoid, quad
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "originais/src"))
from isdbd_regime.profiles import (
    Geometry, ProfileCoefficients, anchored_pair_basis,
    build_profile, evaluate_profile_acceptance,
)
from isdbd_regime.isd import isd_log_trapezoid, isd_direct

H, L, A, EPS, S_N, S_W, D_BULK = 4.0, 8.0, 1.0, 1e-5, 0.58, 0.68, 0.95
NODES = (3201, 6401, 12801, 25601)
NUMERIC_GUARD = 1e-10


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def frame_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.17g",
                 compression={"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None)


def verify_inputs():
    hashes = json.loads((ROOT / "evidencias/entradas_sha256.json").read_text())
    for rel, expected in hashes.items():
        observed = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"Entrada preservada alterada: {rel}")


def geometry(field, sigma):
    if field == "G":
        return Geometry(H, 1.15, 2.0, sigma, 0.40, 0.29)
    return Geometry(H, 1.70, 2.20, sigma, 0.40, 0.40)


def tail_sum(z, geom):
    return (anchored_pair_basis(z, H, geom.mu_inner_nm, geom.sigma_inner_nm)
            + anchored_pair_basis(z, H, geom.mu_outer_nm, geom.sigma_outer_nm))


def central_coefficient(geom, epsilon):
    k0 = float(tail_sum(np.array([0.0, H]), geom)[0])
    q_h = math.exp(-H * H / (2 * geom.sigma_center_nm ** 2))
    return (A - epsilon * k0) / (1 - q_h)


def parameters(sg, sd, epsilon):
    gg, dg = geometry("G", sg), geometry("D", sd)
    return dict(
        a_center=central_coefficient(gg, epsilon), a_inner=epsilon, a_outer=epsilon,
        mu_g_inner_nm=gg.mu_inner_nm, mu_g_outer_nm=gg.mu_outer_nm,
        sigma_g_center_nm=sg, sigma_g_inner_nm=gg.sigma_inner_nm,
        sigma_g_outer_nm=gg.sigma_outer_nm,
        b_center=-central_coefficient(dg, epsilon), b_inner=-epsilon, b_outer=-epsilon,
        mu_d_inner_nm=dg.mu_inner_nm, mu_d_outer_nm=dg.mu_outer_nm,
        sigma_d_center_nm=sd, sigma_d_inner_nm=dg.sigma_inner_nm,
        sigma_d_outer_nm=dg.sigma_outer_nm, D_bulk_nm2_ns=D_BULK,
    )


def original_profile(z, p):
    return build_profile(
        z, geometry("G", p["sigma_g_center_nm"]),
        geometry("D", p["sigma_d_center_nm"]),
        ProfileCoefficients(p["a_center"], p["a_inner"], p["a_outer"]),
        ProfileCoefficients(p["b_center"], p["b_inner"], p["b_outer"]), D_BULK,
    )


def f_sigma(z, sigma):
    qh = math.exp(-H * H / (2 * sigma * sigma))
    return (math.exp(-z * z / (2 * sigma * sigma)) - qh) / (1 - qh)


def scalars(z, name, p=None):
    if p is not None:
        # Avaliação pelas funções originais, não por um modelo reimplementado.
        prof = original_profile(np.array([z, H]), p)
        return float(prof.g[0]), float(-prof.ell[0])
    b = math.cos(math.pi * z / L) ** 2
    return (A * b, A * b ** 4) if name.endswith("A") else (A * b ** 4, A * b)


def quadrature_reference(name, p=None):
    means, errs = {}, {}
    for key in ("MG", "MD", "MGD"):
        def integrand(z):
            g, h = scalars(z, name, p)
            return math.exp(g if key == "MG" else h if key == "MD" else g + h)
        val, err = quad(integrand, -H, H, epsabs=1e-12, epsrel=1e-12, limit=200)
        means[key], errs[key] = val / L, err / L
    ag, ad = math.log(means["MG"]), math.log(means["MD"])
    total = math.log(means["MGD"])
    interaction = total - ag - ad
    return dict(
        profile_id=name, AG=ag, AD=ad, I=interaction, CG=ag + interaction / 2,
        CD=ad + interaction / 2, deltaC=ag-ad, rho=ag-ad,
        max_g_analytic=A, minus_min_ell_analytic=A, d_min_analytic=math.exp(-A),
        eta_analytic=A/math.log(10), R0_ns_nm=L/D_BULK,
        RG_ns_nm=L/D_BULK*means["MG"], RD_ns_nm=L/D_BULK*means["MD"],
        RGD_ns_nm=L/D_BULK*means["MGD"], P_nm_ns=D_BULK/L/means["MGD"],
        P_cm_s=100*D_BULK/L/means["MGD"],
        wG=math.exp(ag-A), wD=math.exp(ad-A), **means,
        quad_estimated_error_MG=errs["MG"], quad_estimated_error_MD=errs["MD"],
        quad_estimated_error_MGD=errs["MGD"],
        quad_estimated_error_delta=errs["MG"]/(means["MG"]-errs["MG"])
                                  + errs["MD"]/(means["MD"]-errs["MD"]),
    )


def original_metric(historico):
    columns = ["sobol_index", "AG", "AD", "g_max", "eta", "deltaC", "rho", "log10_P_cm_s"]
    projection = ROOT / "dados/projecao_historica.csv.gz"
    if historico:
        src = Path(historico).resolve()
        df = pd.read_csv(src, usecols=columns).sort_values("sobol_index", kind="mergesort").reset_index(drop=True)
        frame_csv(df[columns], projection)
        write_json(ROOT / "evidencias/fonte_projecao.json", {
            "historical_csv_name": src.name,
            "historical_csv_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "columns": columns, "N": len(df), "modified_ensemble": False,
            "input_parser": "pandas.read_csv padrão do analisador histórico; projeção serializada com 17 dígitos",
        })
    else:
        df = pd.read_csv(projection, float_precision="round_trip").sort_values("sobol_index", kind="mergesort").reset_index(drop=True)
    assert len(df) == 56926 and df["sobol_index"].is_unique
    g, eta, dc = (df[k].to_numpy(float) for k in ["g_max", "eta", "deltaC"])
    assert np.all(dc != 0)
    xr, yr = float(np.ptp(g)), float(np.ptp(eta))
    xy = np.column_stack([(g-g.min())/xr, (eta-eta.min())/yr])
    it, id_ = np.where(dc > 0)[0], np.where(dc < 0)[0]
    dist, loc = cKDTree(xy[id_]).query(xy[it], k=1, workers=1)
    order = np.argsort(dist, kind="mergesort")[:100]
    rows = []
    for rank, k in enumerate(order, 1):
        i, j = int(it[k]), int(id_[int(loc[k])])
        rr = dict(rank=rank, normalized_proxy_distance=float(dist[k]),
                  thermo_sobol=int(df.iloc[i].sobol_index), diffusive_sobol=int(df.iloc[j].sobol_index))
        for old, new in [("g_max", "g_max"), ("eta", "eta"), ("deltaC", "deltaC"),
                         ("rho", "rho"), ("AG", "AG"), ("AD", "AD"), ("log10_P_cm_s", "log10P")]:
            rr["thermo_"+new], rr["diffusive_"+new] = float(df.iloc[i][old]), float(df.iloc[j][old])
        rr["abs_rho_contrast"] = abs(rr["thermo_rho"]-rr["diffusive_rho"])
        rr["thermo_proxy_margin"] = rr["thermo_g_max"]-math.log(10)*rr["thermo_eta"]
        rr["diffusive_proxy_margin"] = rr["diffusive_g_max"]-math.log(10)*rr["diffusive_eta"]
        rows.append(rr)
    observed = pd.DataFrame(rows)
    frame_csv(observed, ROOT / "dados/pares_proximos_recalculados.csv")
    historical = pd.read_csv(ROOT / "originais/pares_proximos_historicos.csv")
    assert np.array_equal(observed.thermo_sobol, historical.thermo_sobol)
    assert np.array_equal(observed.diffusive_sobol, historical.diffusive_sobol)
    max_diffs = {c: float(np.max(np.abs(observed[c]-historical[c])))
                 for c in historical if c not in {"rank", "thermo_sobol", "diffusive_sobol"}}
    assert max(max_diffs.values()) < 1e-12
    first = rows[0]
    direct_dist = math.hypot((first["thermo_g_max"]-first["diffusive_g_max"])/xr,
                            (first["thermo_eta"]-first["diffusive_eta"])/yr)
    assert abs(direct_dist-first["normalized_proxy_distance"]) < 1e-15
    write_json(ROOT / "evidencias/metrica_historica.json", dict(
        metric="Euclidiana em coordenadas min–max, com alcance de cada eixo na produção aceita",
        g_min=float(g.min()), g_max=float(g.max()), g_range=xr,
        eta_min=float(eta.min()), eta_max=float(eta.max()), eta_range=yr,
        N_thermo=len(it), N_diffusive=len(id_), nearest_k=1, workers=1,
        order="sobol_index mergesort; distâncias mergesort; 100 vizinhos de candidatos termodinâmicos",
        unique_unordered_pairs_top100=int(observed[["thermo_sobol", "diffusive_sobol"]].drop_duplicates().shape[0]),
        minimum_pair=first, direct_distance=direct_dist,
        historical_comparison_max_abs_by_column=max_diffs,
        interpretation="Proximidade no ensemble finito; não é igualdade dos extremos nem prova de não-identificabilidade.",
    ))
    return first


def draw_figure(frame, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "svg.fonttype": "none"})
    blue, orange = "#176B9B", "#C35D1C"
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6), constrained_layout=True)
    z = frame["z_nm"]
    pa, pb = results["original_A"], results["original_B"]
    ax = axes[0, 0]
    ax.plot(z, frame.g_A, color=blue, label=r"$g_A$ (larga)")
    ax.plot(z, frame.g_B, color=orange, ls="--", label=r"$g_B$ (estreita)")
    ax.set(title="A  Energia livre reduzida", ylabel=r"$g(z)$", ylim=(-.04, 1.08))
    ax.scatter([0], [A], color="black", s=18, zorder=5)
    ax.annotate(r"$\max g_A=\max g_B=1$", xy=(0, 1), xytext=(.4, .89), fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax = axes[0, 1]
    ax.plot(z, frame.d_A, color=blue, label=r"$d_A$ (estreita)")
    ax.plot(z, frame.d_B, color=orange, ls="--", label=r"$d_B$ (larga)")
    ax.set(title="B  Difusividade reduzida", ylabel=r"$d(z)$", ylim=(.30, 1.05))
    ax.scatter([0], [math.exp(-A)], color="black", s=18, zorder=5)
    ax.text(.97, .1, r"$\min d_A=\min d_B=e^{-1}$", transform=ax.transAxes, ha="right", fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    for ax, name, color, letter in [(axes[0, 2], "A", blue, "C"), (axes[1, 0], "B", orange, "D")]:
        ax.plot(z, frame["exp_g_"+name], color=color, label=r"$e^{g_"+name+r"}$")
        ax.plot(z, frame["inverse_d_"+name], color=color, ls="--", label=r"$1/d_"+name+r"$")
        ax.fill_between(z, frame["exp_g_"+name], frame["inverse_d_"+name], color=color, alpha=.17)
        ax.set(title=f"{letter}  Integrandas do perfil {name}", ylabel="Integranda adimensional", ylim=(.94, 2.85))
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax = axes[1, 1]
    ax.axhline(0, color="#888888", lw=.8)
    ax.plot(z, frame.cum_difference_A, color=blue, label="A")
    ax.plot(z, frame.cum_difference_B, color=orange, ls="--", label="B")
    ax.set(title="E  Diferença acumulada", ylabel=r"$L^{-1}\int_{-H}^{z}(e^g-1/d)\,ds$")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(.02, .04, "O sinal final é o sinal de ΔC.", transform=ax.transAxes, fontsize=9)
    ax = axes[1, 2]
    ax.plot([.21, .30], [.21, .30], color="#666666", lw=1, label=r"$A_G=A_D$")
    ax.scatter([pa["AG"], pb["AG"]], [pa["AD"], pb["AD"]], c=[blue, orange], s=65, zorder=5)
    ax.annotate(f"A: ΔC = +{pa['deltaC']:.7f}", (pa["AG"], pa["AD"]), xytext=(-87, -18),
                textcoords="offset points", fontsize=9, color=blue)
    ax.annotate(f"B: ΔC = {pb['deltaC']:.7f}", (pb["AG"], pb["AD"]), xytext=(5, 8),
                textcoords="offset points", fontsize=9, color=orange)
    ax.set(title="F  Contribuições integradas", xlabel=r"$A_G$", ylabel=r"$A_D$", xlim=(.21, .30), ylim=(.21, .30))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    for index, ax in enumerate(axes.flat):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if index != 5:
            ax.set_xlim(-H, H)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_xlabel("z (nm)")
    fig.suptitle("Família original: mesmos extremos, classes operacionais opostas", fontsize=15)
    fig.savefig(ROOT / "figuras/CONTRAEXEMPLO_FAMILIA_ORIGINAL.png", dpi=300)
    fig.savefig(ROOT / "figuras/CONTRAEXEMPLO_FAMILIA_ORIGINAL.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historico", help="CSV completo original, apenas na primeira extração")
    args = parser.parse_args()
    verify_inputs()
    bounds = pd.read_csv(ROOT / "originais/PARAMETROS_17.csv")
    assert len(bounds) == 17
    policy = yaml.safe_load((ROOT / "originais/config/r1d2b_production_coefficient_policy.yaml").read_text())
    limits = policy["profile_envelope"] if "profile_envelope" in policy else None
    if limits is None:
        limits = next(v for v in policy.values() if isinstance(v, dict) and {"g_min", "g_max", "d_min", "d_max"} <= v.keys())
    limits = {k: limits[k] for k in ["g_min", "g_max", "d_min", "d_max"]}
    cases = {
        "central_A": parameters(S_W, S_N, 0), "central_B": parameters(S_N, S_W, 0),
        "original_A": parameters(S_W, S_N, EPS), "original_B": parameters(S_N, S_W, EPS),
        "auxiliar_A": None, "auxiliar_B": None,
    }
    write_json(ROOT / "dados/parametros_perfis.json", dict(
        H_nm=H, L_nm=L, a=A, epsilon=EPS, sigma_narrow_nm=S_N, sigma_wide_nm=S_W,
        D_bulk_nm2_ns=D_BULK, profile_parameters={k: v for k, v in cases.items() if v is not None},
        origin="Perfis novos para prova; não são linhas Sobol da produção nem alteram suas contagens",
        auxiliary_definition="b=cos²(pi*z/L); g_A=a*b, h_A=a*b^4; g_B=a*b^4, h_B=a*b",
    ))
    param_rows, checks, refs, grid_rows = [], {}, {}, []
    main_arrays = {}
    for name, p in cases.items():
        if p is not None:
            inside = []
            for row in bounds.to_dict("records"):
                value, lo, hi = p[row["csv_field"]], float(row["low"]), float(row["high"])
                in_box = lo <= value <= hi
                strict = lo < value < hi
                assert in_box
                inside.append(strict)
                param_rows.append(dict(profile_id=name, field=row["csv_field"], value=value,
                                       unit=row["unit"], low=lo, high=hi, u=(value-lo)/(hi-lo),
                                       strict_interior=strict))
            if name.startswith("original"):
                assert all(inside) and all(abs(p[k]) > 0 for k in ["a_center", "a_inner", "a_outer", "b_center", "b_inner", "b_outer"])
            margin_rows = []
            for field in ["G", "D"]:
                geom = geometry(field, p["sigma_g_center_nm" if field == "G" else "sigma_d_center_nm"])
                coefficient = p["a_center"] if field == "G" else -p["b_center"]
                epsilon = p["a_inner"] if field == "G" else -p["b_inner"]
                central_bound = coefficient/geom.sigma_center_nm**2*math.exp(-geom.mu_outer_nm**2/(2*geom.sigma_center_nm**2))
                paired_bound = epsilon*(2/geom.sigma_inner_nm**2+2/geom.sigma_outer_nm**2)
                assert central_bound > paired_bound >= 0
                margin_rows.append(dict(field=field, m_nm=geom.mu_outer_nm,
                                       central_derivative_lower_coefficient=central_bound,
                                       paired_derivative_upper_coefficient=paired_bound,
                                       monotonic_margin=central_bound-paired_bound))
        ref = quadrature_reference(name, p)
        refs[name] = ref
        assert ref["deltaC"] > 0 if name.endswith("A") else ref["deltaC"] < 0
        residuals, acceptance, maximum_grid_error = {}, [], 0.0
        for n in NODES:
            z = np.linspace(-H, H, n)
            if p is not None:
                prof = original_profile(z, p)
                g, h, d = prof.g, -prof.ell, prof.d_ratio
                filt = evaluate_profile_acceptance(prof, **limits)
                assert filt["accepted"]
                acceptance.append(dict(n_nodes=n, **filt))
            else:
                b = np.cos(math.pi*z/L)**2
                g, h = (A*b, A*b**4) if name.endswith("A") else (A*b**4, A*b)
                d = np.exp(-h)
            d_abs = D_BULK*d
            r0 = isd_log_trapezoid(z, np.zeros_like(z), np.full_like(z, D_BULK))
            rg = isd_log_trapezoid(z, g, np.full_like(z, D_BULK))
            rd = isd_log_trapezoid(z, np.zeros_like(z), d_abs)
            rgd = isd_log_trapezoid(z, g, d_abs)
            direct = isd_direct(z, g, d_abs)
            ag = rg.log_resistance-r0.log_resistance
            ad = rd.log_resistance-r0.log_resistance
            dc = ag-ad
            ip = rgd.log_resistance-rg.log_resistance-rd.log_resistance+r0.log_resistance
            diff = max(abs(ag-ref["AG"]), abs(ad-ref["AD"]), abs(dc-ref["deltaC"]),
                       abs(ip-ref["I"]), abs(rgd.permeability_cm_per_s-ref["P_cm_s"])/ref["P_cm_s"])
            maximum_grid_error = max(maximum_grid_error, diff)
            assert diff < NUMERIC_GUARD
            grid_rows.append(dict(profile_id=name, n_nodes=n, AG=ag, AD=ad, deltaC=dc, I=ip,
                                  error_AG=abs(ag-ref["AG"]), error_AD=abs(ad-ref["AD"]),
                                  error_delta=abs(dc-ref["deltaC"]), error_I=abs(ip-ref["I"]),
                                  relative_error_P=abs(rgd.permeability_cm_per_s-ref["P_cm_s"])/ref["P_cm_s"],
                                  log_vs_direct_relative_R=abs(rgd.resistance_ns_per_nm-direct.resistance_ns_per_nm)/direct.resistance_ns_per_nm))
            peak_res = max(abs(float(g.max())-A), abs(float(h.max())-A), abs(float(d.min())-math.exp(-A)))
            assert peak_res < 5e-15
            residuals[str(n)] = dict(peak_residual=peak_res,
                                    g_edges_max_abs=float(np.max(np.abs(g[[0, -1]]))),
                                    h_edges_max_abs=float(np.max(np.abs(h[[0, -1]]))))
            if name.startswith("original") and n == 3201:
                letter = name[-1]
                if "z_nm" not in main_arrays:
                    main_arrays["z_nm"] = z
                main_arrays.update({"g_"+letter:g, "h_"+letter:h, "d_"+letter:d,
                                    "exp_g_"+letter:np.exp(g), "inverse_d_"+letter:np.exp(h),
                                    "cum_difference_"+letter:cumulative_trapezoid(np.exp(g)-np.exp(h), z, initial=0)/L})
        checks[name] = dict(in_family=p is not None, strict_all_17=p is not None and all(inside),
                           analytic_monotonic_checks=margin_rows if p is not None else [],
                           acceptance_original_filter=acceptance, extrema_rounding=residuals,
                           maximum_grid_vs_quad_difference=maximum_grid_error)
    analytic_delta_lower = 1/4800
    perturbation_delta_bound = 16*EPS
    point_gap = f_sigma(.70, S_W)-f_sigma(.65, S_N)
    assert point_gap > .05 and analytic_delta_lower > perturbation_delta_bound
    analytic_margin = analytic_delta_lower-perturbation_delta_bound
    for name in ["original_A", "original_B"]:
        assert abs(refs[name]["deltaC"]) > analytic_margin
    checks["proof_bounds"] = dict(
        interval_abs_z_nm=[.65, .70], endpoint_gap=point_gap,
        pure_delta_strict_lower_bound=analytic_delta_lower,
        per_field_uniform_perturbation_upper_bound=8*EPS,
        delta_perturbation_upper_bound=perturbation_delta_bound,
        perturbed_abs_delta_strict_lower_bound=analytic_margin,
        note="Limite analítico conservador; estimativas QUADPACK e malha não o substituem",
    )
    # Na subfamília central, a troca é exata também para RGD e a permeabilidade.
    assert abs(refs["central_A"]["deltaC"]+refs["central_B"]["deltaC"]) < 2e-15
    assert refs["central_A"]["RGD_ns_nm"] == refs["central_B"]["RGD_ns_nm"]
    frame_csv(pd.DataFrame(param_rows), ROOT / "dados/parametros_17_perfis.csv")
    frame_csv(pd.DataFrame(refs.values()), ROOT / "dados/resistencias_contribuicoes.csv")
    frame_csv(pd.DataFrame(grid_rows), ROOT / "dados/convergencia.csv")
    profile_frame = pd.DataFrame(main_arrays)
    frame_csv(profile_frame, ROOT / "dados/perfis_original_AB.csv.gz")
    write_json(ROOT / "evidencias/verificacao_perfis.json", checks)
    write_json(ROOT / "evidencias/resultados_referencia.json", refs)
    first = original_metric(args.historico)
    draw_figure(profile_frame, refs)
    versions = {pkg:importlib.metadata.version(pkg) for pkg in ["numpy", "scipy", "pandas", "matplotlib", "PyYAML", "Pillow"]}
    write_json(ROOT / "evidencias/ambiente.json", dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), packages=versions))
    (ROOT / "requirements.txt").write_text("\n".join(f"{k}=={v}" for k,v in versions.items())+"\n")
    print(json.dumps(dict(status="PASS", original_A_delta=refs["original_A"]["deltaC"],
                          original_B_delta=refs["original_B"]["deltaC"],
                          analytic_abs_delta_lower=analytic_margin, nearest_historical_pair=first),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
