"""Campos originais, quadratura refinada e solução direta da equação FP."""
from pathlib import Path
from decimal import Decimal, localcontext
import hashlib
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.linalg import solve_banded

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "originais/src"))
from isdbd_regime.profiles import Geometry, ProfileCoefficients, build_profile
from isdbd_regime.isd import isd_log_trapezoid
from isdbd_regime.fokker_planck import bernoulli, steady_dirichlet_sg

PROTOCOL = json.loads((ROOT / "PROTOCOLO_ETAPA5.json").read_text())
FIELDS = pd.read_csv(ROOT / "originais/PARAMETROS_17.csv").csv_field.tolist()
METRICS = ["AG", "AD", "interaction", "CG", "CD", "deltaC", "P_cm_s",
           "g_min", "g_max", "d_min", "d_max", "proxy_margin", "rho"]
H, L = 4., 8.


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_inputs():
    for name, sha in json.loads((ROOT / "evidencias/entradas_sha256.json").read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name


def geometries(theta):
    p = np.asarray(theta, float)
    gg = Geometry(H, p[3], p[4], p[5], p[6], p[7])
    dg = Geometry(H, p[11], p[12], p[13], p[14], p[15])
    return gg, dg, ProfileCoefficients(*p[:3]), ProfileCoefficients(*p[8:11]), p[16]


def source_profile(theta, n):
    z = np.linspace(-H, H, n)
    profile = build_profile(z, *geometries(theta))
    return z, profile


def fields_batch(theta, z, derivatives=False):
    theta = np.atleast_2d(theta)
    z = np.atleast_2d(z)
    v = lambda i: theta[:, i, None]
    def field(c, ci, co, mi, mo, sc, si, so):
        qc = np.exp(-z*z/(2*sc*sc))
        qci = np.exp(-H*H/(2*sc*sc))
        value = c*(qc-qci)
        derivative = -c*z/(sc*sc)*qc if derivatives else None
        for coef, mu, sigma in [(ci, mi, si), (co, mo, so)]:
            qm = np.exp(-(z-mu)**2/(2*sigma*sigma))
            qp = np.exp(-(z+mu)**2/(2*sigma*sigma))
            edge = np.exp(-(H-mu)**2/(2*sigma*sigma))+np.exp(-(H+mu)**2/(2*sigma*sigma))
            value += coef*(qm+qp-edge)
            if derivatives:
                derivative += coef*(-(z-mu)*qm/(sigma*sigma)-(z+mu)*qp/(sigma*sigma))
        return value, derivative
    g, gp = field(*(v(i) for i in range(8)))
    ell, ep = field(*(v(i) for i in range(8,16)))
    return g, ell, gp, ep


def metrics_from_logs(ag, ad, total, theta, gmin, gmax, dmin, dmax):
    interaction = total-ag-ad
    y = -np.log(dmin)
    margin = gmax-y
    return dict(AG=ag, AD=ad, interaction=interaction, CG=ag+interaction/2,
                CD=ad+interaction/2, deltaC=ag-ad,
                P_cm_s=100*np.asarray(theta)[:,16]/L*np.exp(-total),
                g_min=gmin, g_max=gmax, d_min=dmin, d_max=dmax,
                proxy_margin=margin, rho=(ag-gmax)-(ad-y))


def integrate_batch(theta, n, batch=32):
    theta = np.atleast_2d(theta)
    out = {key:np.empty(len(theta)) for key in METRICS}
    z = np.linspace(-H, H, n)
    weights = np.zeros(n)
    weights[[0,-1]] = math.log(.5)
    normalizer = math.log(n-1)
    for start in range(0,len(theta),batch):
        stop = min(start+batch,len(theta))
        p = theta[start:stop]
        g,ell,_,_ = fields_batch(p,z)
        assert np.all(np.isfinite(g)) and np.all(np.isfinite(ell))
        d = np.exp(ell)
        ag = logsumexp(g+weights,axis=1)-normalizer
        ad = logsumexp(-ell+weights,axis=1)-normalizer
        total = logsumexp(g-ell+weights,axis=1)-normalizer
        metrics = metrics_from_logs(ag,ad,total,p,g.min(axis=1),g.max(axis=1),
                                    d.min(axis=1),d.max(axis=1))
        for key in out:out[key][start:stop] = metrics[key]
    return out


def source_metrics(theta, n):
    z, p = source_profile(theta,n)
    zeros = np.zeros(n)
    dbulk = float(theta[16])
    flat = np.full(n,dbulk)
    r0 = isd_log_trapezoid(z,zeros,flat)
    rg = isd_log_trapezoid(z,p.g,flat)
    rd = isd_log_trapezoid(z,zeros,p.D_nm2_ns)
    rgd = isd_log_trapezoid(z,p.g,p.D_nm2_ns)
    result = metrics_from_logs(np.array([rg.log_resistance-r0.log_resistance]),
                              np.array([rd.log_resistance-r0.log_resistance]),
                              np.array([rgd.log_resistance-r0.log_resistance]),
                              np.atleast_2d(theta),np.array([p.g.min()]),np.array([p.g.max()]),
                              np.array([p.d_ratio.min()]),np.array([p.d_ratio.max()]))
    return {key:float(value[0]) for key,value in result.items()}


def scalar_fields(theta,z):
    p = theta
    def field(c,ci,co,mi,mo,sc,si,so):
        result = c*(math.exp(-z*z/(2*sc*sc))-math.exp(-H*H/(2*sc*sc)))
        for coef,mu,sigma in [(ci,mi,si),(co,mo,so)]:
            result += coef*(math.exp(-(z-mu)**2/(2*sigma*sigma))+
                            math.exp(-(z+mu)**2/(2*sigma*sigma))-
                            math.exp(-(H-mu)**2/(2*sigma*sigma))-
                            math.exp(-(H+mu)**2/(2*sigma*sigma)))
        return result
    return field(*p[:8]),field(*p[8:16])


def adaptive_quad(theta, eps=1e-11):
    theta = np.asarray(theta,float)
    z,p = source_profile(theta,3201)
    shifts = [float(p.g.max()),float((-p.ell).max()),float((p.g-p.ell).max())]
    knots = sorted(set([0.,float(theta[3]),float(theta[4]),float(theta[11]),float(theta[12]),H]))
    logs, relerrors, evaluations, messages = [], [], [], []
    for kind,shift in enumerate(shifts):
        def integrand(x):
            g,ell = scalar_fields(theta,x)
            exponent = [g,-ell,g-ell][kind]
            return math.exp(exponent-shift)
        result = quad(integrand,0,H,epsabs=eps,epsrel=eps,points=knots,
                      limit=300,full_output=1)
        value,error,info = result[:3]
        assert value>0 and math.isfinite(value)
        logs.append(math.log(value/H)+shift)
        relerrors.append(float(error/value))
        evaluations.append(int(info["neval"]))
        messages.append(str(result[3]) if len(result)>3 else "")
    ag,ad,total = logs
    interaction = total-ag-ad
    return dict(AG=ag,AD=ad,interaction=interaction,CG=ag+interaction/2,
                CD=ad+interaction/2,deltaC=ag-ad,P_cm_s=100*theta[16]/L*math.exp(-total),
                quad_AG_error_estimate=relerrors[0],quad_AD_error_estimate=relerrors[1],
                quad_total_log_error_estimate=relerrors[2],
                quad_delta_error_estimate=relerrors[0]+relerrors[1],
                quad_neval=evaluations,quad_messages=messages)


def fp_direct(theta, n, precision=60, floating_diagnostic=False):
    """Resolve A q=b pela eliminação tridiagonal, não pela soma de resistências."""
    z,p = source_profile(theta,n)
    h = float(z[1]-z[0])
    dg = np.diff(p.g)
    face_D = np.sqrt(p.D_nm2_ns[:-1]*p.D_nm2_ns[1:])
    conductance = (face_D/h)*bernoulli(dg)*np.exp(-p.g[:-1])
    assert np.all(conductance>0) and np.all(np.isfinite(conductance))
    diagnostic = {}
    if floating_diagnostic:
        matrix = np.zeros((3,n-2))
        matrix[1] = conductance[:-1]+conductance[1:]
        matrix[0,1:] = -conductance[1:-1]
        matrix[2,:-1] = -conductance[1:-1]
        rhs = np.zeros(n-2);rhs[0] = conductance[0]
        try:
            q64 = np.r_[1.,solve_banded((1,1),matrix,rhs),0.]
            j64 = conductance*(q64[:-1]-q64[1:])
            diagnostic.update(float64_flux_mean=float(j64.mean()),
                              float64_flux_min=float(j64.min()),float64_flux_max=float(j64.max()),
                              float64_q_min=float(q64.min()),float64_q_max=float(q64.max()),
                              float64_error=None)
        except Exception as exc:
            diagnostic["float64_error"] = type(exc).__name__+": "+str(exc)
    with localcontext() as ctx:
        ctx.prec = int(precision)
        c = [Decimal.from_float(float(x)) for x in conductance]
        # A diagonal é construída em Decimal para não perder o termo menor.
        upper,solution_rhs = [],[]
        denominator = c[0]+c[1]
        upper.append(-c[1]/denominator)
        solution_rhs.append(c[0]/denominator)
        for k in range(1,n-2):
            lower = -c[k]
            denominator = c[k]+c[k+1]-lower*upper[k-1]
            assert denominator>0
            upper.append(-c[k+1]/denominator)
            solution_rhs.append(-lower*solution_rhs[k-1]/denominator)
        q = [Decimal(0)]*n
        q[0] = Decimal(1)
        q[n-2] = solution_rhs[-1]
        for k in range(n-4,-1,-1):
            q[k+1] = solution_rhs[k]-upper[k]*q[k+2]
        fluxes = [c[k]*(q[k]-q[k+1]) for k in range(n-1)]
        flux = sum(fluxes)/Decimal(n-1)
        assert flux>0 and min(q)>=0 and max(q)<=1
        nonuniformity = max(abs(x-flux) for x in fluxes)/flux
        residuals = []
        for k in range(1,n-1):
            a = c[k-1]*(q[k]-q[k-1])
            b = c[k]*(q[k]-q[k+1])
            scale = abs(c[k-1]*q[k])+abs(c[k-1]*q[k-1])+abs(c[k]*q[k])+abs(c[k]*q[k+1])
            residuals.append(abs(a+b)/scale if scale else Decimal(0))
        backward = max(residuals)
        result = dict(grid_N=n,precision_digits=precision,P_nm_ns=float(flux),P_cm_s=100*float(flux),
                      relative_flux_nonuniformity=float(nonuniformity),
                      maximum_scaled_equation_residual=float(backward),
                      q_min=float(min(q)),q_max=float(max(q)),
                      p_min=float(min(float(x)*math.exp(-float(g)) for x,g in zip(q,p.g))),
                      boundary_error=0.,conductance_ratio=float(conductance.max()/conductance.min()))
    network = steady_dirichlet_sg(z,p.g,p.D_nm2_ns)
    result["relative_direct_vs_network"] = abs(result["P_nm_ns"]/network.permeability_nm_per_ns-1.)
    result.update(diagnostic)
    return result
