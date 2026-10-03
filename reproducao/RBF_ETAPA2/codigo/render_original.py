from pathlib import Path
import hashlib,json,math,os
import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image

ROOT=Path(os.environ["ISDBD_ROOT"])
OUT=Path(os.environ["ISDBD_JOB"])

DATA=ROOT/"data/model/r2d2bfix2_production_ensemble.csv.gz"
POLICY=ROOT/"config/r2d2gfix1_figure_refinement_policy_amendment.yaml"

policy=yaml.safe_load(POLICY.read_text(encoding="utf-8"))
assert policy["status"]=="FIGURE_REFINEMENT_POLICY_AMENDMENT_FROZEN"
assert policy["source_dataset"] if "source_dataset" in policy else True

mpl.rcParams.update({
    "font.family":"DejaVu Sans",
    "font.size":9.2,
    "axes.titlesize":10.6,
    "axes.labelsize":9.6,
    "xtick.labelsize":8.1,
    "ytick.labelsize":8.1,
    "legend.fontsize":7.4,
    "pdf.fonttype":42,
    "ps.fonttype":42,
    "axes.linewidth":0.8,
    "savefig.facecolor":"white",
})

df=pd.read_csv(DATA,compression="gzip").sort_values("sobol_index",kind="mergesort").reset_index(drop=True)

required=[
    "sobol_index","AG","AD","g_max","eta","deltaC","rho","log10_P_cm_s",
    "evaluation_grid_N","numerical_refinement_applied"
]
missing=[c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"missing required columns: {missing}")

if len(df)!=56926 or df["sobol_index"].nunique()!=56926:
    raise RuntimeError("production point-count/uniqueness failure")
if not np.all(np.diff(df["sobol_index"].to_numpy(dtype=np.int64))>0):
    raise RuntimeError("Sobol order is not strictly ascending")

for c in ["AG","AD","g_max","eta","deltaC","rho","log10_P_cm_s"]:
    if not np.all(np.isfinite(df[c].to_numpy(dtype=float))):
        raise RuntimeError(f"nonfinite primary field: {c}")

ref=df.loc[df["sobol_index"]==36312]
if len(ref)!=1 or int(ref.iloc[0]["evaluation_grid_N"])!=12801:
    raise RuntimeError("refined Sobol 36312 record invalid")
if int(np.sum(df["evaluation_grid_N"].to_numpy(dtype=int)==12801))!=1:
    raise RuntimeError("unexpected refined-record count")

AG=df["AG"].to_numpy(dtype=float)
AD=df["AD"].to_numpy(dtype=float)
gmax=df["g_max"].to_numpy(dtype=float)
eta=df["eta"].to_numpy(dtype=float)
dc=df["deltaC"].to_numpy(dtype=float)
rho=df["rho"].to_numpy(dtype=float)
logP=df["log10_P_cm_s"].to_numpy(dtype=float)

id1=float(np.max(np.abs(dc-(AG-AD))))
id2=float(np.max(np.abs(dc-(gmax-math.log(10.0)*eta+rho))))
if id1>1e-12 or id2>1e-12:
    raise RuntimeError(f"architecture identity failure: {id1}, {id2}")

# Frozen R2_D2G-FIX1 evidence.
proxy_accuracy=float(np.mean(np.sign(gmax-math.log(10.0)*eta)==np.sign(dc)))
proxy_mismatch_fraction=1.0-proxy_accuracy
if abs(proxy_accuracy-0.92318097178793523)>1e-15:
    raise RuntimeError(f"proxy accuracy mismatch: {proxy_accuracy}")
if abs(proxy_mismatch_fraction-0.07681902821206478)>1e-15:
    raise RuntimeError(f"proxy mismatch mismatch: {proxy_mismatch_fraction}")

dc_min=float(np.min(dc)); dc_max=float(np.max(dc))
rho_min=float(np.min(rho)); rho_max=float(np.max(rho))
if abs(dc_min-(-3.11290289628037))>1e-12 or abs(dc_max-11.1046366306192)>1e-11:
    raise RuntimeError("DeltaC frozen range mismatch")
if abs(rho_min-(-3.1759830905598))>1e-12 or abs(rho_max-0.98515116929905)>1e-12:
    raise RuntimeError("rho frozen range mismatch")

inset_lo=-0.0369240244576066
inset_hi=1.84862759843489
near=np.abs(dc)<=0.10
near_N=int(np.sum(near))
if near_N!=985:
    raise RuntimeError(f"near-boundary N={near_N} !=985")

ranges={
    "AG":[float(np.min(AG)),float(np.max(AG))],
    "AD":[float(np.min(AD)),float(np.max(AD))],
    "g_max":[float(np.min(gmax)),float(np.max(gmax))],
    "eta":[float(np.min(eta)),float(np.max(eta))],
    "deltaC":[dc_min,dc_max],
    "rho":[rho_min,rho_max],
    "log10_P_cm_s":[float(np.min(logP)),float(np.max(logP))],
}

norm_dc=TwoSlopeNorm(vmin=dc_min,vcenter=0.0,vmax=dc_max)
norm_rho=TwoSlopeNorm(vmin=rho_min,vcenter=0.0,vmax=rho_max)

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def render(pdf_path,png_path):
    fig,axs=plt.subplots(1,3,figsize=(18.0,6.2),constrained_layout=True)

    # Panel A: full exact control plane.
    ax=axs[0]
    sc=ax.scatter(
        AG,AD,c=logP,s=2.0,marker="o",linewidths=0,alpha=0.75,
        cmap="viridis",rasterized=True,
        vmin=ranges["log10_P_cm_s"][0],vmax=ranges["log10_P_cm_s"][1]
    )
    lo=max(ranges["AG"][0],ranges["AD"][0])
    hi=min(ranges["AG"][1],ranges["AD"][1])
    ax.plot([lo,hi],[lo,hi],linestyle="-",linewidth=1.15,
            label=r"Exact boundary: $A_G=A_D$")
    ax.set_xlabel(r"Thermodynamic burden, $A_G=\ln(R_G/R_0)$")
    ax.set_ylabel(r"Diffusional burden, $A_D=\ln(R_D/R_0)$")
    ax.set_title("A  Exact control plane",loc="left",fontweight="bold")
    ax.legend(loc="upper left",frameon=False)
    ax.text(0.97,0.05,"Thermodynamic control\n$A_G>A_D$",
            transform=ax.transAxes,ha="right",va="bottom")
    ax.text(0.03,0.95,"Diffusional control\n$A_G<A_D$",
            transform=ax.transAxes,ha="left",va="top")
    cb=fig.colorbar(sc,ax=ax,pad=0.02)
    cb.set_label(r"$\log_{10} P$ (cm s$^{-1}$)")

    # Required equal-scale near-boundary inset.
    iax=ax.inset_axes([0.54,0.50,0.42,0.40])
    iax.scatter(
        AG[near],AD[near],c=logP[near],s=5.0,marker="o",
        linewidths=0,alpha=0.82,cmap="viridis",rasterized=True,
        vmin=ranges["log10_P_cm_s"][0],vmax=ranges["log10_P_cm_s"][1]
    )
    iax.plot([inset_lo,inset_hi],[inset_lo,inset_hi],linestyle="-",linewidth=0.9)
    iax.set_xlim(inset_lo,inset_hi)
    iax.set_ylim(inset_lo,inset_hi)
    iax.set_aspect("equal",adjustable="box")
    iax.set_title(r"Near boundary: $|\Delta C|\leq0.10$ ($n=985$)",fontsize=6.2,pad=2.0)
    iax.tick_params(axis="both",labelsize=5.4,length=2.0,pad=1.0)
    iax.set_xlabel(r"$A_G$",fontsize=5.8,labelpad=1)
    iax.set_ylabel(r"$A_D$",fontsize=5.8,labelpad=1)

    # Panel B: intuitive proxy with full observed asymmetric signed range.
    ax=axs[1]
    sc=ax.scatter(
        gmax,eta,c=dc,s=2.0,marker="o",linewidths=0,alpha=0.75,
        cmap="coolwarm",norm=norm_dc,rasterized=True
    )
    gx=np.linspace(max(0.0,ranges["g_max"][0]),ranges["g_max"][1],300)
    gy=gx/math.log(10.0)
    m=(gy>=ranges["eta"][0]) & (gy<=ranges["eta"][1])
    ax.plot(gx[m],gy[m],linestyle="--",linewidth=1.05,
            label=r"Amplitude-only reference: $\eta=\max(g)/\ln 10$")
    ax.set_xlabel(r"Maximum free-energy amplitude, $\max(g)$")
    ax.set_ylabel(r"Maximum diffusional suppression, $\eta=-\log_{10} d_{\min}$")
    ax.set_title("B  Intuitive amplitude projection",loc="left",fontweight="bold")
    ax.legend(loc="upper right",frameon=False)
    ax.text(
        0.97,0.05,
        "Amplitude proxy\n92.32% agreement\n7.68% mismatch",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25",facecolor="white",edgecolor="0.55",alpha=0.88)
    )
    cb=fig.colorbar(sc,ax=ax,pad=0.02)
    cb.set_label(r"$\Delta C=C_G-C_D$")

    # Panel C: spatial-shape correction with full observed asymmetric signed range.
    ax=axs[2]
    sc=ax.scatter(
        gmax,eta,c=rho,s=2.0,marker="o",linewidths=0,alpha=0.75,
        cmap="coolwarm",norm=norm_rho,rasterized=True
    )
    ax.set_xlabel(r"Maximum free-energy amplitude, $\max(g)$")
    ax.set_ylabel(r"Maximum diffusional suppression, $\eta=-\log_{10} d_{\min}$")
    ax.set_title("C  Spatial-shape correction",loc="left",fontweight="bold")
    ax.text(
        0.03,0.05,
        r"$\Delta C=\max(g)-\ln(10)\eta+\rho$",
        transform=ax.transAxes,ha="left",va="bottom",fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25",facecolor="white",edgecolor="0.55",alpha=0.88)
    )
    cb=fig.colorbar(sc,ax=ax,pad=0.02)
    cb.set_label(r"$\rho=\ln(w_G/w_D)$")

    fig.suptitle(
        "Thermodynamic and diffusional control of passive membrane permeability",
        fontsize=12.2,fontweight="bold"
    )

    pdf_meta={
        "Title":"ISD-BD refined regime maps",
        "Author":"ISD-BD-Regime",
        "Subject":"Refined production maps from frozen deterministic ensemble",
        "Keywords":"permeability free-energy diffusivity resistance regime",
        "Creator":"ISD-BD-Regime",
        "Producer":"Matplotlib",
        "CreationDate":None,
        "ModDate":None,
    }
    fig.savefig(pdf_path,format="pdf",dpi=300,metadata=pdf_meta)
    fig.savefig(png_path,format="png",dpi=600,metadata={"Software":"ISD-BD-Regime"})
    plt.close(fig)

pdf1=OUT/"R2_D2H_REFINED_PRODUCTION_MAPS.pdf"
png1=OUT/"R2_D2H_REFINED_PRODUCTION_MAPS_600dpi.png"
pdf2=OUT/"R2_D2H_REFINED_PRODUCTION_MAPS_repeat.pdf"
png2=OUT/"R2_D2H_REFINED_PRODUCTION_MAPS_600dpi_repeat.png"

render(pdf1,png1)
render(pdf2,png2)

hashes={
    "pdf_first":sha256(pdf1),
    "pdf_repeat":sha256(pdf2),
    "png_first":sha256(png1),
    "png_repeat":sha256(png2),
}
pdf_deterministic=(hashes["pdf_first"]==hashes["pdf_repeat"])
png_deterministic=(hashes["png_first"]==hashes["png_repeat"])
if not pdf_deterministic or not png_deterministic:
    raise RuntimeError(f"non-deterministic refined figure hashes: {hashes}")

pdf2.unlink(); png2.unlink()

with Image.open(png1) as im:
    width,height=im.size
    mode=im.mode
    arr=np.asarray(im.convert("RGB"),dtype=np.uint8)

if width<9000 or height<3000:
    raise RuntimeError(f"600dpi PNG unexpectedly small: {width}x{height}")
pixel_std=float(np.std(arr))
black_fraction=float(np.mean(np.all(arr<5,axis=2)))
white_fraction=float(np.mean(np.all(arr>250,axis=2)))
if pixel_std<5.0 or black_fraction>0.20 or white_fraction>0.995:
    raise RuntimeError("raster sanity failure")

metrics={
  "mission_id":"ISDBD-R2-D2H-RENDER-AND-AUDIT-REFINED-PRODUCTION-MAPS-001",
  "source_dataset":"data/model/r2d2bfix2_production_ensemble.csv.gz",
  "source_dataset_sha256":sha256(DATA),
  "policy":"config/r2d2gfix1_figure_refinement_policy_amendment.yaml",
  "point_count":len(df),
  "sobol_unique_count":int(df["sobol_index"].nunique()),
  "refined_record":{"sobol_index":36312,"evaluation_grid_N":12801},
  "identity_checks":{
    "max_abs_deltaC_minus_AG_minus_AD":id1,
    "max_abs_deltaC_minus_proxy_shape":id2,
    "tolerance":1e-12
  },
  "panel_A":{
    "main":"AG_vs_AD_log10P_exact_boundary",
    "near_boundary_inset":{
      "definition":"abs_deltaC_le_0p10",
      "N":near_N,
      "xlim":[inset_lo,inset_hi],
      "ylim":[inset_lo,inset_hi],
      "equal_coordinate_scale":True,
      "same_log10P_normalization_as_main":True
    }
  },
  "panel_B":{
    "normalization":"TwoSlopeNorm",
    "vmin":dc_min,"vcenter":0.0,"vmax":dc_max,
    "complete_observed_range":True,
    "proxy_accuracy":proxy_accuracy,
    "proxy_mismatch_fraction":proxy_mismatch_fraction
  },
  "panel_C":{
    "normalization":"TwoSlopeNorm",
    "vmin":rho_min,"vcenter":0.0,"vmax":rho_max,
    "complete_observed_range":True,
    "identity_annotation":"DeltaC=max(g)-ln(10)*eta+rho"
  },
  "rendering":{
    "all_points_preserved":True,
    "draw_order":"ascending_sobol_index",
    "interpolation":False,
    "smoothing":False,
    "extrapolation":False,
    "aggregation":False,
    "percentile_clipping":False,
    "png_dpi":600,
    "figure_inches":[18.0,6.2]
  },
  "outputs":{
    "pdf":"figures/R2_D2H_REFINED_PRODUCTION_MAPS.pdf",
    "png":"figures/R2_D2H_REFINED_PRODUCTION_MAPS_600dpi.png",
    "pdf_sha256":hashes["pdf_first"],
    "png_sha256":hashes["png_first"],
    "pdf_bytes":pdf1.stat().st_size,
    "png_bytes":png1.stat().st_size,
    "png_width_px":width,
    "png_height_px":height,
    "png_mode":mode
  },
  "determinism":{
    "pdf_repeat_hash_match":bool(pdf_deterministic),
    "png_repeat_hash_match":bool(png_deterministic)
  },
  "raster_sanity":{
    "pixel_std":pixel_std,
    "black_fraction":black_fraction,
    "white_fraction":white_fraction
  },
  "map_audit_pass":bool(
      len(df)==56926 and near_N==985 and id1<=1e-12 and id2<=1e-12
      and pdf_deterministic and png_deterministic
      and abs(dc_min-(-3.11290289628037))<=1e-12
      and abs(dc_max-11.1046366306192)<=1e-11
      and abs(rho_min-(-3.1759830905598))<=1e-12
      and abs(rho_max-0.98515116929905)<=1e-12
  ),
  "model_compute_executed":False,
  "production_dataset_modified":False,
  "old_canonical_figure_modified":False,
  "MFPT_computed":False,
  "novelty_claim_made":False
}
(OUT/"r2d2h_refined_production_map_audit.json").write_text(
    json.dumps(metrics,indent=2,sort_keys=True)+"\n",encoding="utf-8"
)

print("__RESULT_BEGIN__")
print(f"point_count={len(df)}")
print(f"sobol_unique_count={df['sobol_index'].nunique()}")
print("refined_sobol_36312_present=YES")
print("refined_sobol_36312_grid_N=12801")
print(f"near_boundary_inset_N={near_N}")
print(f"inset_lo={inset_lo:.12e}")
print(f"inset_hi={inset_hi:.12e}")
print(f"proxy_accuracy={proxy_accuracy:.12e}")
print(f"proxy_mismatch_fraction={proxy_mismatch_fraction:.12e}")
print(f"deltaC_vmin={dc_min:.12e}")
print(f"deltaC_vcenter={0.0:.12e}")
print(f"deltaC_vmax={dc_max:.12e}")
print(f"rho_vmin={rho_min:.12e}")
print(f"rho_vcenter={0.0:.12e}")
print(f"rho_vmax={rho_max:.12e}")
print(f"max_identity_deltaC_AG_AD={id1:.12e}")
print(f"max_identity_proxy_shape={id2:.12e}")
print(f"pdf_sha256={hashes['pdf_first']}")
print(f"png_sha256={hashes['png_first']}")
print(f"pdf_deterministic={'YES' if pdf_deterministic else 'NO'}")
print(f"png_deterministic={'YES' if png_deterministic else 'NO'}")
print(f"pdf_bytes={pdf1.stat().st_size}")
print(f"png_bytes={png1.stat().st_size}")
print(f"png_dimensions={width}x{height}")
print(f"png_pixel_std={pixel_std:.12e}")
print(f"png_black_fraction={black_fraction:.12e}")
print(f"png_white_fraction={white_fraction:.12e}")
print(f"map_audit_pass={'YES' if metrics['map_audit_pass'] else 'NO'}")
print("model_compute_executed=NO")
print("production_dataset_modified=NO")
print("old_canonical_figure_modified=NO")
print("MFPT_computed=NO")
print("novelty_claim_made=NO")
print("__RESULT_END__")

if not metrics["map_audit_pass"]:
    raise SystemExit(7)
