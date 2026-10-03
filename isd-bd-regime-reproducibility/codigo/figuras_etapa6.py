"""Renderização editorial; reutiliza os resultados encerrados nas etapas 3–5.

Não executa ensembles nem trajetórias. Avalia apenas curvas dos exemplos escolhidos
para desenhá-las; as contribuições e as contagens vêm das tabelas preservadas.
"""
from pathlib import Path
import sys, json, hashlib, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullLocator, ScalarFormatter

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"reproducao/RBF_ETAPA5"
BASE4=ROOT/"reproducao/RBF_ETAPA4"
sys.path.insert(0,str(BASE/"originais/src"))
from isdbd_regime.profiles import Geometry,ProfileCoefficients,build_profile

BLUE,ORANGE,GREEN,GRAY="#196b96","#b85521","#407843","#7c858c"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
 "axes.titlesize":9.5,"axes.labelsize":9,"legend.fontsize":8,
 "xtick.labelsize":8.5,"ytick.labelsize":8.5,"pdf.fonttype":42,
 "svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False,
 "savefig.facecolor":"white"})
FMT=FuncFormatter(lambda x,p:f"{x:g}".replace(".",","))
W=16/2.54
evidence={"data_source":"dados/producao_revisao_numerica.csv.gz da etapa 5",
 "production_N":56926,"scientific_campaign_reexecuted":False,
 "display_width_cm":16,"minimum_regular_font_pt":8,
 "figures":{}}

def finish(fig,name,height,source):
    fig.set_size_inches(W,height/2.54)
    for ax in fig.axes:
        if ax.get_xscale()=="linear" and isinstance(ax.xaxis.get_major_formatter(),ScalarFormatter):ax.xaxis.set_major_formatter(FMT)
        if ax.get_yscale()=="linear" and isinstance(ax.yaxis.get_major_formatter(),ScalarFormatter):ax.yaxis.set_major_formatter(FMT)
    for ext in ("pdf","png","svg"):
        kw={"metadata":{"Creator":"Matplotlib; revisão RBF etapa 6","CreationDate":None}} if ext=="pdf" else {}
        buffer=io.BytesIO()
        fig.savefig(buffer,format=ext,dpi=450,**kw)
        data=buffer.getvalue()
        if ext=='pdf':
            import fitz
            assert data.rstrip().endswith(b'%%EOF'),name
            with fitz.open(stream=data,filetype='pdf') as doc:
                assert len(doc)==1 and not doc.is_repaired,name
        path=ROOT/"figuras"/f"{name}.{ext}"
        temporary=path.with_name(path.name+'.writing')
        temporary.write_bytes(data)
        temporary.replace(path)
    evidence["figures"][name]={"width_cm":16,"height_cm":height,"source":source,
          "pdf_sha256":hashlib.sha256((ROOT/"figuras"/f"{name}.pdf").read_bytes()).hexdigest()}
    plt.close(fig)

def profile(row,z):
    g=Geometry(4,row['mu_g_inner_nm'],row['mu_g_outer_nm'],row['sigma_g_center_nm'],row['sigma_g_inner_nm'],row['sigma_g_outer_nm'])
    d=Geometry(4,row['mu_d_inner_nm'],row['mu_d_outer_nm'],row['sigma_d_center_nm'],row['sigma_d_inner_nm'],row['sigma_d_outer_nm'])
    a=ProfileCoefficients(*(row[k] for k in ['a_center','a_inner','a_outer']))
    b=ProfileCoefficients(*(row[k] for k in ['b_center','b_inner','b_outer']))
    return build_profile(z,g,d,a,b,row['D_bulk_nm2_ns'])

def num(x,n=3):return f"{x:.{n}f}".replace(".",",")

def legend_above(ax):
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,borderaxespad=0)
    ax.set_title(ax.get_title(),pad=28)

def main():
    frame=pd.read_csv(BASE/"dados/producao_revisao_numerica.csv.gz",float_precision="round_trip")
    frame['m']=frame.g_max-np.log(10)*frame.eta
    ct=np.sign(frame.deltaC);cp=np.sign(frame.m)
    matrix=[[int(((ct==a)&(cp==b)).sum()) for b in (1,-1)] for a in (1,-1)]
    assert matrix==[[29161,7],[4366,23392]] and len(frame)==56926
    assert np.allclose(frame.deltaC,frame.m+frame.rho,rtol=0,atol=1e-12)
    assert int(frame.stage5_revision_applied.sum())==6
    evidence['matrix_T_D']=matrix
    evidence['corrected_record_N']=6
    fig,ax=plt.subplots(1,2,layout="constrained")
    for cls,col in [(-1,ORANGE),(1,BLUE)]:
        f=frame[ct==cls]
        label="D: contribuição difusiva" if cls<0 else "T: contribuição termodinâmica"
        ax[0].scatter(f.AG,f.AD,s=.85,c=col,alpha=.38,edgecolors='none',rasterized=True,label=label)
        ax[1].scatter(f.g_max,f.eta,s=.85,c=col,alpha=.38,edgecolors='none',rasterized=True)
    ax[0].plot([0,2.5],[0,2.5],c='.2',ls='--',lw=1)
    ax[0].set(xlabel=r"$A_G=\ln\langle e^g\rangle$",ylabel=r"$A_D=\ln\langle1/d\rangle$",title="A  Contribuições integradas",xlim=(-2.4,12),ylim=(-.08,2.6))
    ax[0].legend(loc="upper right",markerscale=5,frameon=True,facecolor='white',edgecolor='none',framealpha=1,handletextpad=.3)
    eta=np.linspace(0,1.05,100)
    ax[1].plot(np.log(10)*eta,eta,c='.2',ls='--',lw=1)
    ax[1].set(xlabel=r"$x=\max g(z)$",ylabel=r"$\eta=-\log_{10}d_{\min}$",title="B  Projeção pelos extremos",xlim=(-.2,13.3),ylim=(-.04,1.1))
    a_visible=frame.AG.between(*ax[0].get_xlim()) & frame.AD.between(*ax[0].get_ylim())
    b_visible=frame.g_max.between(*ax[1].get_xlim()) & frame.eta.between(*ax[1].get_ylim())
    assert a_visible.all() and b_visible.all()
    evidence['classification_panels_visible_N']=[int(a_visible.sum()),int(b_visible.sum())]
    finish(fig,"FIG1_CLASSIFICACAO",7.9,"Etapa 5: vista revisada dos 56.926 originais; todos os pontos.")

    selected=[]
    for actual,pred,name in [(-1,1,"D_para_T"),(1,-1,"T_para_D")]:
        f=frame[(ct==actual)&(cp==pred)].copy()
        # Representante mais próximo das medianas conjuntas de m e rho,
        # em unidades da amplitude interquartil; empate pelo menor ID Sobol.
        cols=['m','rho'];v=f[cols].to_numpy();med=np.median(v,axis=0)
        scale=np.quantile(v,.75,axis=0)-np.quantile(v,.25,axis=0)
        assert np.all(scale>0)
        f['score_representative']=np.square((v-med)/scale).sum(axis=1)
        row=f.sort_values(['score_representative','sobol_index']).iloc[0].copy()
        row['group']=name;row['group_N']=len(f);selected.append(row)
    choices=pd.DataFrame(selected)
    choices.to_csv(ROOT/'dados/perfis_representativos.csv',index=False,float_format='%.17g')
    fig,ax=plt.subplots(2,2,layout='constrained')
    curves=[]
    z=np.linspace(-4,4,3201)
    for j,row in enumerate(selected):
        p=profile(row,z);hg=-p.ell
        a=ax[0,j];a.plot(z,p.g,c=BLUE,lw=1.5,label=r"$g$")
        a.plot(z,hg,c=ORANGE,lw=1.5,label=r"$-\ln d$");a.axhline(0,c='.7',lw=.7)
        title='D → T' if j==0 else 'T → D'
        a.set(title=f"{'A' if j==0 else 'B'}  Integrada → proxy: {title}",xlabel="Posição z (nm)",ylabel="Campos adimensionais")
        a.legend(frameon=False,loc='best',ncol=2)
        a.text(.02,.02,f"Sobol {int(row.sobol_index):,}".replace(',','.'),transform=a.transAxes,fontsize=8)
        ug=np.exp(p.g-row.g_max);ud=np.exp(-p.ell+np.log(row.d_min))
        a=ax[1,j];a.plot(z,ug,c=BLUE,lw=1.5,label=r"$e^{g-x}$");a.plot(z,ud,c=ORANGE,lw=1.5,label=r"$e^{-\ell-y}$")
        a.fill_between(z,0,ug,color=BLUE,alpha=.10);a.fill_between(z,0,ud,color=ORANGE,alpha=.10)
        a.set(xlabel="Posição z (nm)",ylabel="Integrandas normalizadas",ylim=(-.05,1.08),title=f"{'C' if j==0 else 'D'}  Suportes efetivos")
        legend_above(a)
        curves.append(pd.DataFrame({'sobol_index':int(row.sobol_index),'z_nm':z,'g':p.g,'d':p.d_ratio,'G_normalized':ug,'D_normalized':ud}))
    pd.concat(curves).to_csv(ROOT/'dados/curvas_representativas.csv.gz',index=False,float_format='%.17g')
    finish(fig,"FIG2_PERFIS_DISCORDANTES",12.6,"Regra determinística de medianas/IQR dos dois grupos; curvas avaliadas só para desenho.")
    evidence['representatives']=[{'group':r.group,'N':int(r.group_N),'sobol_index':int(r.sobol_index),'m':float(r.m),'rho':float(r.rho),'deltaC':float(r.deltaC)} for r in selected]

    # Figura do resumo de quatro páginas: mapas completos e suportes dos mesmos
    # representantes. Usa as curvas já avaliadas, sem nova quadratura/campanha.
    fig,ax=plt.subplots(2,2,layout='constrained')
    for cls,col in [(-1,ORANGE),(1,BLUE)]:
        f=frame[ct==cls]
        ax[0,0].scatter(f.AG,f.AD,s=.7,c=col,alpha=.38,edgecolors='none',rasterized=True,label='T' if cls>0 else 'D')
        ax[0,1].scatter(f.g_max,f.eta,s=.7,c=col,alpha=.38,edgecolors='none',rasterized=True)
    ax[0,0].plot([0,2.5],[0,2.5],c='.2',ls='--',lw=1)
    ax[0,0].set(xlabel=r'$A_G$',ylabel=r'$A_D$',title='A  Classificação integrada',xlim=(-2.4,12),ylim=(-.08,2.6))
    ax[0,0].legend(loc='upper right',markerscale=5,ncol=2,frameon=True,facecolor='white',edgecolor='none',framealpha=1)
    ax[0,1].plot(np.log(10)*eta,eta,c='.2',ls='--',lw=1)
    ax[0,1].set(xlabel=r'$x=\max g$',ylabel=r'$\eta=-\log_{10}d_{\min}$',title='B  Projeção pelos extremos',xlim=(-.2,13.3),ylim=(-.04,1.1))
    for j,curve in enumerate(curves):
        a=ax[1,j]
        a.plot(curve.z_nm,curve.G_normalized,c=BLUE,lw=1.4,label=r'$e^{g-x}$')
        a.plot(curve.z_nm,curve.D_normalized,c=ORANGE,lw=1.4,label=r'$e^{-\ell-y}$')
        a.fill_between(curve.z_nm,0,curve.G_normalized,color=BLUE,alpha=.10)
        a.fill_between(curve.z_nm,0,curve.D_normalized,color=ORANGE,alpha=.10)
        a.set(xlabel='Posição z (nm)',ylabel='Integrandas normalizadas',ylim=(-.05,1.08),
              title='C  D → T: Sobol 24.214' if j==0 else 'D  T → D: Sobol 37.905')
        legend_above(a)
    finish(fig,'FIG0_RESUMO',9.0,'Mapas completos e suportes dos mesmos representantes; síntese gráfica para o resumo expandido.')

    params=json.loads((ROOT/'originais/parametros_perfis.json').read_text())['profile_parameters']
    values=pd.read_csv(ROOT/'originais/resistencias_contribuicoes.csv',float_precision='round_trip')
    fig,ax=plt.subplots(2,2,layout='constrained')
    curves=[]
    for i,name in enumerate(['original_A','original_B']):
        p=profile(params[name],z);lab='A' if i==0 else 'B';ls='-' if i==0 else '--'
        ax[0,0].plot(z,p.g,c=BLUE,ls=ls,lw=1.6,label=f'Perfil {lab}')
        ax[0,1].plot(z,p.d_ratio,c=ORANGE,ls=ls,lw=1.6,label=f'Perfil {lab}')
        a=ax[1,i];a.plot(z,np.exp(p.g),c=BLUE,lw=1.5,label=r'$e^g$')
        a.plot(z,1/p.d_ratio,c=ORANGE,lw=1.5,label=r'$1/d$')
        a.set(title=f"{'C' if i==0 else 'D'}  Integrandas: perfil {lab}",xlabel='Posição z (nm)',ylabel='Integrandas isoladas',ylim=(.92,2.9))
        legend_above(a)
        # Valores da etapa 3, sem nova integração.
        vv=values[values.profile_id==name].iloc[0]
        a.text(.03,.88,f"ΔC = {num(vv.deltaC,6)}",transform=a.transAxes,fontsize=8,
              bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
        curves.append(pd.DataFrame({'profile_id':name,'z_nm':z,'g':p.g,'d':p.d_ratio}))
    ax[0,0].set(title='A  Mesmo máximo energético',xlabel='Posição z (nm)',ylabel=r'$g(z)$',ylim=(-.04,1.08))
    ax[0,1].set(title='B  Mesmo mínimo difusivo',xlabel='Posição z (nm)',ylabel=r'$d(z)$',ylim=(.33,1.05))
    ax[0,0].legend(frameon=True,ncol=1,loc='upper left',facecolor='white',edgecolor='none',framealpha=1)
    ax[0,1].legend(frameon=True,ncol=1,loc='lower left',facecolor='white',edgecolor='none',framealpha=1)
    ax[0,0].axhline(1,c='.6',ls=':',lw=.8)
    ax[0,1].axhline(np.exp(-1),c='.6',ls=':',lw=.8)
    pd.concat(curves).to_csv(ROOT/'dados/curvas_construtivas.csv.gz',index=False,float_format='%.17g')
    finish(fig,'FIG3_EXTREMOS_IGUAIS',11.5,'Etapa 3: original_A/original_B, distintos da produção. Igualdade analítica; quadraturas prévias reutilizadas.')

    fig,ax=plt.subplots(1,2,layout='constrained',gridspec_kw={'width_ratios':[1.4,1]})
    groups=[(1,1,'T / T',BLUE,'o'),(-1,-1,'D / D',GRAY,'o'),(-1,1,'D / T',ORANGE,'o'),(1,-1,'T / D',GREEN,'D')]
    for a,b,label,c,marker in groups:
        f=frame[(ct==a)&(cp==b)]
        ax[0].scatter(f.m,f.rho,s=28 if len(f)==7 else 1.1,c=c,marker=marker,alpha=1 if len(f)==7 else .4,
                     edgecolors='none',rasterized=len(f)>7,label=label)
    xx=np.linspace(-2.5,3.5,100)
    ax[0].plot(xx,-xx,c='.2',ls='--',lw=1)
    ax[0].set(title='A  Correção de suporte',xlabel=r'$m=x-y$',ylabel=r'$\rho=\ln(w_G/w_D)$',xlim=(-2.5,3.5),ylim=(-3,1))
    ax[0].legend(frameon=False,ncol=2,loc='upper right',markerscale=2)
    # Extremos que saem do zoom A são declarados; B inclui todos os grupos.
    visible=frame.m.between(-2.5,3.5)&frame.rho.between(-3,1)
    evidence['support_panel_A_visible_N']=int(visible.sum())
    ratios=[np.exp(frame.loc[(ct==a)&(cp==b),'rho'].to_numpy()) for a,b,*_ in groups]
    bp=ax[1].boxplot(ratios,positions=np.arange(4),orientation='horizontal',widths=.5,whis=(0,100),showfliers=False,
      tick_labels=['T / T','D / D','D / T','T / D'],patch_artist=True)
    for patch,gp in zip(bp['boxes'],groups):patch.set_facecolor(gp[3]);patch.set_alpha(.35)
    ax[1].set(title='B  Razão de suportes',xlabel=r'$w_G/w_D$');ax[1].set_xscale('log');ax[1].axvline(1,c='.5',ls=':',lw=.8)
    finish(fig,'FIGA1_SUPORTES',7.9,'Vista revisada: A é zoom declarado; B mínimo/máximo, quartis e mediana de todos os grupos.')

    dom=pd.read_csv(BASE4/'dados/resumo_dominio.csv')
    dom=dom[dom.kind.isin(['historical_reference','domain_remap'])]
    samp=pd.read_csv(BASE4/'dados/resumo_amostragem.csv')
    fig,axes=plt.subplots(2,1,layout='constrained',gridspec_kw={'height_ratios':[1.15,1]})
    labs=['Referência','Amplitudes G × 0,80','Amplitudes ln d × 0,80','Larguras G: metade central',
          'Larguras D: metade inferior','Larguras D: metade superior','Posições: metade central']
    yy=np.arange(7)
    for key,lab,c,m in [('mismatch_pct','Global',BLUE,'o'),('diffusive_error_pct','Dentro da classe D',ORANGE,'D'),('thermo_error_pct','Dentro da classe T',GREEN,'^')]:
        axes[0].scatter(dom[key],yy,c=c,marker=m,s=28,label=lab)
    axes[0].set_yticks(yy,labs);axes[0].invert_yaxis();axes[0].set(xlabel='Erro (%)',xlim=(-.8,20),title='A  Domínios remapeados')
    axes[0].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25),columnspacing=.7,handletextpad=.3)
    axes[0].set_title('A  Domínios remapeados',pad=32);axes[0].grid(axis='x',alpha=.2)
    n=samp[samp.kind=='sampling_nested'];s=samp[samp.kind=='sampling_scramble']
    axes[1].plot(np.log2(n.candidate_N),n.mismatch_pct,'o-',c=BLUE,label='Sobol não embaralhado')
    axes[1].scatter(16+np.array([-.12,-.04,.04,.12]),s.mismatch_pct,c=ORANGE,marker='D',s=30,label='4 embaralhamentos em 2¹⁶')
    axes[1].set_xticks([14,15,16,17],[r'$2^{14}$',r'$2^{15}$',r'$2^{16}$',r'$2^{17}$'])
    axes[1].set(xlabel='Candidatos por desenho',ylabel='Erro global (%)',title='B  Tamanho e amostragem',ylim=(7.4,7.95))
    axes[1].legend(frameon=False,loc='lower right');axes[1].grid(alpha=.2)
    finish(fig,'FIGA2_SENSIBILIDADES',14.0,'Etapa 4: resumo_dominio.csv e resumo_amostragem.csv; sem regenerar cenários.')

    summary=json.loads((BASE/'evidencias/consolidacao.json').read_text())
    fp=json.loads((BASE/'evidencias/resumo_fp.json').read_text())
    bd=json.loads((BASE/'evidencias/resumo_bd.json').read_text())
    precision=pd.read_csv(BASE/'dados/precisao_todos_originais.csv.gz')
    q=np.log10(abs(precision.final_deltaC)/precision.deltaC_error_estimate)
    fig,ax=plt.subplots(2,2,layout='constrained')
    bins=np.linspace(5.5,13.7,38)
    ax[0,0].hist(q,bins=bins,color=BLUE,alpha=.8,label='Todos os originais')
    ax[0,0].hist(q[precision.boundary_0p10_original],bins=bins,color=ORANGE,label='985 próximos da fronteira')
    ax[0,0].set(title='A  Margem / erro estimado',xlabel=r'$\log_{10}(|\Delta C|/\widehat{\delta\Delta C})$',ylabel='Número de perfis')
    ax[0,0].legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),borderaxespad=0,fontsize=8)
    ax[0,0].set_title('A  Margem / erro estimado',pad=36,y=1.0)
    ns=[r['grid_N'] for r in summary['quad_meshes']]
    ax[0,1].loglog(ns,[r['max_relative_P_vs_QUAD'] for r in summary['quad_meshes']],'o-',c=BLUE,label='ISD / QUADPACK (96)')
    ax[0,1].loglog(ns,[r['max_relative_P'] for r in fp['rows']],'s-',c=ORANGE,label='FP / ISD (185)')
    ax[0,1].axhline(1e-7,c=BLUE,ls=':',lw=.8);ax[0,1].axhline(5e-5,c=ORANGE,ls=':',lw=.8)
    ax[0,1].set(title='B  Erros em subconjuntos',xlabel='Nós da malha',ylabel='Maior erro relativo de P')
    ax[0,1].set_xticks(ns,['3.201','6.401','12.801','25.601']);ax[0,1].xaxis.set_minor_locator(NullLocator())
    ax[0,1].tick_params(axis='x',rotation=25)
    ax[0,1].legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),borderaxespad=0,fontsize=8)
    ax[0,1].set_title('B  Erros em subconjuntos',pad=36,y=1.0)
    labels=['EM · 0,002','Milstein · 0,002','EM · 0,001']
    for i,r in enumerate(bd['operational_boundary_margins']):
        ax[1,0].errorbar(r['deltaC_BD'],i,xerr=1.96*r['deltaC_MC_SE'],fmt='o',capsize=3,c=[BLUE,ORANGE,GREEN][i])
    ax[1,0].axvline(0,c='.4',ls='--',lw=1);ax[1,0].set_yticks(range(3),labels)
    ax[1,0].set(title='C  Controle junto à fronteira',xlabel=r'$\Delta C_{\rm BD}=\ln(P_D/P_G)$',ylabel='Integrador · passo (ns)',xlim=(-.06,.065),ylim=(2.5,-.5))
    states=['FLAT','G_ONLY','D_ONLY','GD_BOUNDARY']
    for i,(method,dt) in enumerate([('EM',.002),('MILSTEIN',.002),('EM',.001)]):
        rr=[next(r for r in bd['rows'] if r['case_id']==state and r['method']==method and r['dt_ns']==dt) for state in states]
        ax[1,1].errorbar(np.arange(4)+(i-1)*.17,[r['P_BD_nm_ns']/r['P_FP_nm_ns'] for r in rr],yerr=[1.96*r['P_BD_SE_nm_ns']/r['P_FP_nm_ns'] for r in rr],fmt='o',capsize=2,c=[BLUE,ORANGE,GREEN][i],label=labels[i])
    ax[1,1].axhline(1,c='.4',ls='--',lw=1);ax[1,1].set_xticks(range(4),['Plano','G','D','GD'])
    ax[1,1].set(title='D  Permeabilidade Browniana',ylabel=r'$P_{\rm BD}/P_{\rm FP}$',xlabel='Estado contrafactual',ylim=(.925,1.075))
    ax[1,1].legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),borderaxespad=0,fontsize=8)
    ax[1,1].set_title('D  Permeabilidade Browniana',pad=47,y=1.0)
    finish(fig,'FIGA3_PRECISAO',13.6,'Etapa 5: erros estimados de integração, comparações de métodos e intervalos condicionais MC95.')

    hist=json.loads((BASE/'originais/controles_bd/r2c2c_fix2_scope_evidence.json').read_text())['P4_NARROW_OUTER']
    runs=['EM_dt0.002','EM_dt0.001','MILSTEIN_dt0.002','MILSTEIN_dt0.001']
    fig,ax=plt.subplots(layout='constrained');xx=np.arange(4)
    for off,key,lab,c in [(-.17,'right_event_counts','Saídas à direita',BLUE),(.17,'unresolved_counts','Trajetórias não resolvidas',ORANGE)]:
        vals=[hist[key][r] for r in runs];ax.bar(xx+off,vals,width=.34,color=c,label=lab)
        for x,v in zip(xx+off,vals):ax.text(x,v+2,str(v),ha='center',fontsize=9)
    ax.set_xticks(xx,['EM\n0,002','EM\n0,001','Milstein\n0,002','Milstein\n0,001'])
    ax.set(title='Controle histórico P4: horizonte de 1.500 ns',xlabel='Integrador e passo (ns)',ylabel='Contagens disponíveis',ylim=(0,185))
    ax.legend(frameon=False,loc='upper right',ncol=2)
    finish(fig,'FIGA4_CENSURA_P4',7.0,'Resumo histórico, sem denominadores completos: contagens não são proporções ou permeabilidade.')
    (ROOT/'evidencias/figuras_etapa6.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'matrix':matrix,'representatives':evidence['representatives'],'figures':len(evidence['figures'])},ensure_ascii=False))

if __name__=='__main__':main()
