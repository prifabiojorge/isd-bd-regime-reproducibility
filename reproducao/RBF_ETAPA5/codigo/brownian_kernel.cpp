// Itô EM/Milstein, primeira saída e ponte local. Fonte fornecido para reprodução.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

struct Model {
    double a[3], b[3], mug[3], mud[3], sg[3], sd[3], edgeg[3], edged[3], bulk;
    explicit Model(const double* p) {
        for(int j=0;j<3;++j){a[j]=p[j];b[j]=p[8+j];}
        mug[0]=mud[0]=0.;mug[1]=p[3];mug[2]=p[4];mud[1]=p[11];mud[2]=p[12];
        for(int j=0;j<3;++j){sg[j]=p[5+j];sd[j]=p[13+j];}
        bulk=p[16];
        for(int j=0;j<3;++j){
            edgeg[j]=std::exp(-(4.-mug[j])*(4.-mug[j])/(2.*sg[j]*sg[j]));
            edged[j]=std::exp(-(4.-mud[j])*(4.-mud[j])/(2.*sd[j]*sd[j]));
            if(j){
                edgeg[j]+=std::exp(-(4.+mug[j])*(4.+mug[j])/(2.*sg[j]*sg[j]));
                edged[j]+=std::exp(-(4.+mud[j])*(4.+mud[j])/(2.*sd[j]*sd[j]));
            }
        }
    }
    void field(double z,const double* co,const double* mu,const double* sigma,
               const double* edge,double& value,double& derivative) const {
        value=derivative=0.;
        for(int j=0;j<3;++j){
            if(co[j]==0.)continue;
            const double inv=1./(sigma[j]*sigma[j]);
            const double zm=z-mu[j];
            const double em=std::exp(-.5*zm*zm*inv);
            double q=em,dq=-zm*inv*em;
            if(j){
                const double zp=z+mu[j],ep=std::exp(-.5*zp*zp*inv);
                q+=ep;dq-=zp*inv*ep;
            }
            value+=co[j]*(q-edge[j]);derivative+=co[j]*dq;
        }
    }
    void fields(double z,double* out) const {
        double g,gp,ell,ep;
        field(z,a,mug,sg,edgeg,g,gp);field(z,b,mud,sd,edged,ell,ep);
        const double D=bulk*std::exp(ell),Dp=D*ep;
        out[0]=g;out[1]=ell;out[2]=D;out[3]=gp;out[4]=ep;out[5]=Dp;out[6]=Dp-D*gp;
    }
};

extern "C" void fields_many(const double* p,const double* z,int n,double* out){
    Model model(p);
    for(int i=0;i<n;++i)model.fields(z[i],out+7*i);
}

extern "C" double raw_step(const double* p,double z,double dt,double xi,int milstein){
    Model model(p);double f[7];model.fields(z,f);
    return z+f[6]*dt+std::sqrt(2.*f[2]*dt)*xi+
           (milstein ? .5*f[5]*dt*(xi*xi-1.):0.);
}

extern "C" int simulate(const double* p,const double* starts,int groups,int paths,
                        double dt,std::uint64_t seed,int milstein,int max_steps,
                        std::int64_t* counts,std::int64_t* diagnostics,double* max_prob_sum){
    Model model(p);std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal(0.,1.);
    std::uniform_real_distribution<double> uniform(0.,1.);
    for(int i=0;i<groups*4;++i)counts[i]=0;
    for(int i=0;i<5;++i)diagnostics[i]=0;
    *max_prob_sum=0.;
    for(int group=0;group<groups;++group){
        for(int path=0;path<paths;++path){
            double z=starts[group];int outcome=0,steps=0;
            for(steps=1;steps<=max_steps;++steps){
                double f[7];model.fields(z,f);
                const double xi=normal(engine);
                const double trial=z+f[6]*dt+std::sqrt(2.*f[2]*dt)*xi+
                      (milstein ? .5*f[5]*dt*(xi*xi-1.):0.);
                if(!std::isfinite(trial) || f[2]<=0.)return 2;
                if(trial<=-4.)outcome=-1;
                else if(trial>=4.)outcome=1;
                else {
                    // Exata no controle plano. Aproximação local para campos não planos.
                    double pl=std::exp(-((z+4.)*(trial+4.))/(f[2]*dt));
                    double pr=std::exp(-((4.-z)*(4.-trial))/(f[2]*dt));
                    const double sum=pl+pr;
                    *max_prob_sum=std::max(*max_prob_sum,sum);
                    if(sum>1.){pl/=sum;pr/=sum;++diagnostics[4];}
                    const double u=uniform(engine);
                    if(u<pl){outcome=-1;++diagnostics[0];}
                    else if(u<pl+pr){outcome=1;++diagnostics[1];}
                }
                if(outcome)break;
                z=trial;
            }
            const int used=std::min(steps,max_steps);
            diagnostics[2]=std::max(diagnostics[2],std::int64_t(used));
            diagnostics[3]+=used;
            ++counts[group*4];
            if(outcome==1)++counts[group*4+1];
            else if(outcome==-1)++counts[group*4+2];
            else ++counts[group*4+3];
        }
    }
    return 0;
}
