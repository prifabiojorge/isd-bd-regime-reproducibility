"""Figuras científicas estáticas a partir das evidências consolidadas."""
from numerica import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter,NullLocator


def main():
    summary=json.loads((ROOT/"evidencias/consolidacao.json").read_text())
    fp=json.loads((ROOT/"evidencias/resumo_fp.json").read_text())
    bd=json.loads((ROOT/"evidencias/resumo_bd.json").read_text())
    precision=pd.read_csv(ROOT/"dados/precisao_todos_originais.csv.gz")
    ratios=np.log10(abs(precision.final_deltaC)/precision.deltaC_error_estimate)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"svg.fonttype":"none","savefig.facecolor":"white"})
    blue,orange,green="#226e96","#b95719","#34866f"
    fig,ax=plt.subplots(2,2,figsize=(11.4,8.4),layout="constrained")
    bins=np.linspace(5.5,13.7,42)
    assert ratios.min()>=bins[0] and ratios.max()<=bins[-1]
    ax[0,0].hist(ratios,bins=bins,color=blue,alpha=.85,label="56.926 originais")
    ax[0,0].hist(ratios[precision.boundary_0p10_original],bins=bins,color=orange,
                 label=r"985 com $|\Delta C|\leq0,10$")
    ax[0,0].set(xlabel=r"$\log_{10}(|\Delta C|/\widehat{\delta\Delta C})$",ylabel="Número de perfis",
                 title="A  Margem / erro estimado de integração")
    ax[0,0].legend(frameon=False,fontsize=9)
    ax[0,0].text(.98,.70,"Todos os sinais preservados\nMenor razão: 505.167\nEstimativa sem certificação",
                 transform=ax[0,0].transAxes,fontsize=9,ha="right",
                 bbox=dict(facecolor="white",alpha=.9,edgecolor="none"))
    nq=[r["grid_N"] for r in summary["quad_meshes"]]
    ax[0,1].loglog(nq,[r["max_relative_P_vs_QUAD"] for r in summary["quad_meshes"]],"o-",color=blue,label="Trapezoidal / QUADPACK (96)")
    ax[0,1].loglog(nq,[r["max_relative_P"] for r in fp["rows"]],"s-",color=orange,label="FP direto / ISD (185)")
    ax[0,1].axhline(1e-7,color=blue,ls=":",lw=1,label=r"Critério ISD: $10^{-7}$")
    ax[0,1].axhline(5e-5,color=orange,ls=":",lw=1,label=r"Critério FP: $5\times10^{-5}$")
    ax[0,1].set(xlabel="Nós da malha uniforme",ylabel="Maior erro relativo de P",
                title="B  Convergência em subconjuntos distintos")
    ax[0,1].set_xticks(nq,["3.201","6.401","12.801","25.601"])
    ax[0,1].xaxis.set_minor_locator(NullLocator())
    ax[0,1].xaxis.set_minor_formatter(NullFormatter())
    ax[0,1].legend(frameon=False,fontsize=8,loc="lower left")
    labels=["EM · 0,002 ns","Milstein · 0,002 ns","EM · 0,001 ns"]
    margins=bd["operational_boundary_margins"]
    for i,r in enumerate(margins):
        ax[1,0].errorbar(r["deltaC_BD"],i,xerr=1.96*r["deltaC_MC_SE"],fmt="o",capsize=4,
                         color=[blue,orange,green][i])
    ax[1,0].axvline(0,color=".4",lw=1,ls="--")
    ax[1,0].set(xlim=(-.062,.067),ylim=(-.5,2.5),xlabel=r"$\Delta C_{\rm BD}=\ln(P_D/P_G)$",
                 title="C  Controle numérico próximo da fronteira")
    ax[1,0].set_yticks(range(3),labels)
    ax[1,0].invert_yaxis()
    ax[1,0].text(.01,.015,"Intervalos MC de 95% condicionais; todos incluem zero",
                 transform=ax[1,0].transAxes,fontsize=8)
    states=["FLAT","G_ONLY","D_ONLY","GD_BOUNDARY"]
    for i,(method,dt) in enumerate([("EM",.002),("MILSTEIN",.002),("EM",.001)]):
        rows=[next(r for r in bd["rows"] if r["case_id"]==state and r["method"]==method and r["dt_ns"]==dt)
              for state in states]
        ax[1,1].errorbar(np.arange(4)+(i-1)*.17,[r["P_BD_nm_ns"]/r["P_FP_nm_ns"] for r in rows],
             yerr=[1.96*r["P_BD_SE_nm_ns"]/r["P_FP_nm_ns"] for r in rows],fmt="o",capsize=3,
             color=[blue,orange,green][i],label=labels[i])
    ax[1,1].axhline(1,color=".4",lw=1,ls="--")
    ax[1,1].set(ylim=(.925,1.075),ylabel=r"$P_{\rm BD}/P_{\rm FP}$",title="D  Quarteto contrafactual Browniano")
    ax[1,1].set_xticks(range(4),["Plano","Somente G","Somente D","G e D"])
    ax[1,1].legend(frameon=False,fontsize=8,loc="upper right")
    ax[1,1].text(.01,.02,"32.768 trajetórias por início · 5 inícios por execução\n12 execuções · 1.966.080 trajetórias · censura zero",
                 transform=ax[1,1].transAxes,fontsize=8)
    for ext in ["png","svg"]:fig.savefig(ROOT/"figuras"/f"ETAPA5_PRECISAO_E_FRONTEIRA.{ext}",dpi=190)
    plt.close(fig)
    frame=pd.read_csv(ROOT/"dados/fokker_planck_direto.csv")
    diag=frame[frame.grid_N==6401]
    hist=json.loads((ROOT/"originais/controles_bd/r2c2c_fix2_scope_evidence.json").read_text())["P4_NARROW_OUTER"]
    fig,ax=plt.subplots(1,2,figsize=(11.4,4.1),layout="constrained")
    ax[0].loglog(diag.max_conductance_ratio,np.maximum(diag.float64_P_relative_error,1e-16),"o",ms=3,color=blue,alpha=.65)
    ax[0].set(xlabel="Razão máxima entre condutâncias (4 estados)",ylabel="Erro relativo P float64 / FP direto",
              title="A  Diagnóstico de solução linear em float64")
    ax[0].text(.02,.95,"185 perfis · 6.401 nós\nMaior desvio: 3,052%\nDiagnóstico fora da produção",
                transform=ax[0].transAxes,va="top",fontsize=9)
    runs=["EM_dt0.002","EM_dt0.001","MILSTEIN_dt0.002","MILSTEIN_dt0.001"]
    xs=np.arange(4)
    ax[1].bar(xs-.18,[hist["right_event_counts"][r] for r in runs],width=.36,color=blue,label="Saídas à direita")
    ax[1].bar(xs+.18,[hist["unresolved_counts"][r] for r in runs],width=.36,color=orange,label="Trajetórias censuradas")
    ax[1].set(ylim=(0,195),ylabel="Contagens no resumo histórico",title="B  Cobertura histórica limitada: P4")
    ax[1].set_xticks(xs,["EM\n0,002","EM\n0,001","Milstein\n0,002","Milstein\n0,001"])
    ax[1].set_xlabel("Integrador e passo (ns) · horizonte: 1.500 ns")
    ax[1].legend(frameon=False,fontsize=9,loc="upper left")
    for x,r in zip(xs,runs):
        for offset,key in [(-.18,"right_event_counts"),(.18,"unresolved_counts")]:
            value=hist[key][r];ax[1].text(x+offset,value+3,str(value),ha="center",fontsize=8)
    for ext in ["png","svg"]:fig.savefig(ROOT/"figuras"/f"ETAPA5_LIMITES_DOS_CONTROLES.{ext}",dpi=190)
    plt.close(fig)


if __name__=="__main__":main()
