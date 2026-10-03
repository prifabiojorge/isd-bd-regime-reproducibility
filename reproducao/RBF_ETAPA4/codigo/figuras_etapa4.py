"""Figuras científicas da sensibilidade e dos suportes, sem suavização de pontos."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from motor_sensibilidade import *

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.labelsize":11,
                     "axes.titlesize":12,"svg.fonttype":"none"})
BLUE,ORANGE,GREEN,GRAY="#176B9B","#C35D1C","#267D48","#B5BCC2"


def main():
    dom=pd.read_csv(ROOT/"dados/resumo_dominio.csv")
    dom=dom[dom.kind.isin(["historical_reference","domain_remap"])]
    sample=pd.read_csv(ROOT/"dados/resumo_amostragem.csv")
    cohorts=pd.read_csv(ROOT/"dados/coortes_AG.csv").set_index("cohort")
    labels=["Referência","Amplitudes G × 0,80","Amplitudes ln d × 0,80",
            "Larguras G: metade central","Larguras D: metade inferior",
            "Larguras D: metade superior","Posições: metade central"]
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.8),constrained_layout=True)
    y=np.arange(len(dom))
    ax=axes[0,0]
    for key,lab,color,marker in [("mismatch_pct","Global",BLUE,"o"),
                                 ("diffusive_error_pct","Classe difusiva",ORANGE,"D"),
                                 ("thermo_error_pct","Classe termodinâmica",GREEN,"^")]:
        ax.scatter(dom[key],y,color=color,marker=marker,s=35,label=lab,zorder=4)
    ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.set(xlabel="Erro (%) — denominador de cada série",title="A  Sensibilidade aos parâmetros",xlim=(-.7,20))
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),fontsize=8,frameon=False,ncol=3,
              handletextpad=.4,columnspacing=1.)
    ax.set_title("A  Sensibilidade aos parâmetros",pad=34)
    ax.grid(axis="x",alpha=.2)
    ax=axes[0,1]
    ax.scatter(dom.thermo_fraction_pct,y,color=BLUE,s=35)
    ax.axvline(dom.thermo_fraction_pct.iloc[0],color=GRAY,ls="--",lw=1)
    for pos,value in zip(y,dom.thermo_fraction_pct):
        ax.annotate(f"{value:.2f}%",(value,pos),xytext=(5,-3),textcoords="offset points",fontsize=9)
    ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.set(xlabel="Termodinâmicos entre os aceitos (%)",title="B  Composição do ensemble filtrado",xlim=(46,54.3))
    ax.grid(axis="x",alpha=.2)
    ax.text(.03,.03,"Eixo ampliado; proporções do desenho sintético.",transform=ax.transAxes,fontsize=8)
    ax.set_ylim(6.8,-.3)
    ax=axes[1,0]
    nested=sample[sample.kind=="sampling_nested"].copy()
    powers=np.log2(nested.candidate_N.to_numpy())
    ax.plot(powers,nested.mismatch_pct,color=BLUE,marker="o",lw=1.5,label="Não embaralhado")
    replicas=sample[sample.kind=="sampling_scramble"]
    # Pequenos deslocamentos horizontais servem apenas para distinguir os quatro marcadores em N=2^16.
    xj=16+np.array([-.12,-.04,.04,.12])
    ax.scatter(xj,replicas.mismatch_pct,color=ORANGE,marker="D",s=38,label="4 embaralhamentos em 2¹⁶",zorder=5)
    for xp,r in zip(xj,replicas.itertuples()):
        ax.annotate(str(r.case_id.rsplit("_",1)[1]),(xp,r.mismatch_pct),xytext=(4,4),textcoords="offset points",fontsize=8,color=ORANGE)
    ax.set_xticks([14,15,16,17],[r"$2^{14}$",r"$2^{15}$",r"$2^{16}$",r"$2^{17}$"])
    ax.set(xlabel="Número de candidatos",ylabel="Erro global entre os aceitos (%)",
           title="C  Sensibilidade à amostragem",xlim=(13.8,17.2),ylim=(7.4,7.95))
    ax.legend(loc="lower right",fontsize=9,frameon=False)
    ax.grid(alpha=.2)
    ax=axes[1,1]
    rows=[cohorts.loc["AG_negativo"],cohorts.loc["AG_nao_negativo"]]
    values=[r.diffusive_error_pct for r in rows]
    ax.barh([0,1],values,color=[BLUE,ORANGE],height=.38)
    for yp,r,v in zip([0,1],rows,values):
        ax.text(v+1.5,yp,f"{v:.2f}%\n{int(r.diffusive_as_thermo_N):,}/{int(r.diffusive_N):,}".replace(",","."),va="center",fontsize=9)
    ax.set_yticks([0,1],[r"Difusivos com $A_G<0$",r"Difusivos com $A_G\geq0$"])
    ax.invert_yaxis()
    ax.set(xlabel="Erro dentro de cada subconjunto difusivo (%)",title=r"D  $A_G<0$ não explica toda a assimetria",xlim=(0,110))
    ax.grid(axis="x",alpha=.2)
    ax.text(.03,.03,"Denominadores: 23.952 e 3.806 perfis difusivos.",transform=ax.transAxes,fontsize=8)
    ax.set_ylim(1.45,-.35)
    for ax in axes.flat:ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Etapa 4 — domínio, desenho e denominadores",fontsize=16)
    fig.savefig(ROOT/"figuras/ETAPA4_DOMINIO_AMOSTRAGEM.png",dpi=300)
    fig.savefig(ROOT/"figuras/ETAPA4_DOMINIO_AMOSTRAGEM.svg")
    plt.close(fig)

    base=load_case("referencia_original");valid=base["accepted"]
    wrongd=valid&(base["integrated_class"]==-1)&(base["proxy_class"]==1)
    wrongt=valid&(base["integrated_class"]==1)&(base["proxy_class"]==-1)
    correct=valid&~wrongd&~wrongt
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.8),constrained_layout=True)
    ax=axes[0]
    for mask,color,size,alpha,lab,marker in [
        (correct,GRAY,2,.25,"Concordantes (52.553)","o"),
        (wrongd,ORANGE,5,.45,"Difusivos previstos T (4.366)","o"),
        (wrongt,GREEN,40,1,"Termodinâmicos previstos D (7)","^")]:
        ax.scatter(base["proxy_margin"][mask],base["rho"][mask],c=color,s=size,alpha=alpha,label=lab,marker=marker,rasterized=True)
    lim=(float(base["proxy_margin"][valid].min()),float(base["proxy_margin"][valid].max()))
    xx=np.linspace(*lim,200);ax.plot(xx,-xx,color="#555555",ls="--",lw=1,label=r"$\rho=-m$")
    ax.set(xlabel=r"Margem do proxy $m=\max g-\ln(10)\eta$",ylabel=r"$\rho=\ln(w_G/w_D)$",
           title="A  Correção de suporte na referência",xlim=(lim[0]-.15,lim[1]+.15),
           ylim=(float(base["rho"][valid].min())-.15,float(base["rho"][valid].max())+.15))
    ax.legend(loc="upper right",fontsize=8,frameon=False)
    ax.grid(alpha=.15)
    ax=axes[1]
    groups=[
        ("T previsto T",valid&(base["integrated_class"]==1)&(base["proxy_class"]==1),BLUE),
        ("D previsto D",valid&(base["integrated_class"]==-1)&(base["proxy_class"]==-1),BLUE),
        ("D previsto T",wrongd,ORANGE),("T previsto D",wrongt,GREEN)]
    for y,(name,mask,color) in enumerate(groups):
        ratio=base["wG"][mask]/base["wD"][mask];q=np.quantile(ratio,[0,.25,.5,.75,1])
        ax.plot([q[0],q[4]],[y,y],color=color,alpha=.3,lw=1)
        ax.plot([q[1],q[3]],[y,y],color=color,lw=5,solid_capstyle="butt")
        ax.scatter([q[2]],[y],color=color,s=35,zorder=3)
        ax.annotate(f"mediana {q[2]:.3f}",(q[2],y),xytext=(4,-14),textcoords="offset points",fontsize=8,color=color)
    ax.axvline(1,color="#666666",ls="--",lw=1)
    ax.set_yticks(range(4),[g[0] for g in groups]);ax.invert_yaxis()
    ax.set(xlabel=r"Razão $w_G/w_D$",title="B  Extensão efetiva das integrandas")
    ax.text(.02,.01,"Linha: min–max; segmento: Q1–Q3; ponto: mediana.\nDescrições do ensemble; não intervalos de confiança.",transform=ax.transAxes,fontsize=8)
    ax.set_ylim(3.9,-.6)
    for ax in axes:ax.spines[["top","right"]].set_visible(False)
    fig.savefig(ROOT/"figuras/ETAPA4_SUPORTES_DISCORDANCIAS.png",dpi=300)
    fig.savefig(ROOT/"figuras/ETAPA4_SUPORTES_DISCORDANCIAS.svg")
    plt.close(fig)
    print("Figuras da etapa 4 geradas.")


if __name__=="__main__":main()
