//---------------------------------------------------------------------------
// GpuMom.cpp - MoM dense matrix fill + COCG solve on the GPU
//
// One thread per matrix entry fills the dense complex-symmetric impedance
// matrix Z (same reduced-kernel EFIE quadrature as the CPU path), then a
// diagonally preconditioned COCG (single precision, dense mat-vec) solves
// Z I = V on the GPU. The solution is written into the solver's I vector.
// Falls back to the CPU fill+direct solve on any failure.
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GpuMom.h"
#include "MomSolver.h"
#include "MomSurface.h"
#include "GpuCompute.h"
#include <vector>
#include <complex>
#include <cmath>

#pragma package(smart_init)

// surface RWG GPU path is implemented at the bottom of this file
bool RunGpuMomSurf(MomSurface &, std::string &msg);

static const double C0  = 299792458.0;
static const double MU0 = 1.25663706212e-6;

//---------------------------------------------------------------------------
static const char *MOM_COMMON = R"(
#version 430
layout(local_size_x = 64) in;
vec2 cmul(vec2 a, vec2 b){ return vec2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
vec2 cdiv(vec2 a, vec2 b){ float d=b.x*b.x+b.y*b.y;
    return vec2((a.x*b.x+a.y*b.y)/d, (a.y*b.x-a.x*b.y)/d); }
)";

// dense EFIE matrix fill: one thread per (m,n) entry
static const char *MOM_FILL = R"(
layout(std430, binding=0) buffer BG { vec4 bg[]; };   // 4 vec4 / basis
layout(std430, binding=1) buffer BZ { vec2 Z[]; };
uniform int N;
uniform float kk, wmu, a2;
const float GU[8] = float[8](0.019855071751232,0.101666761293187,0.237233795041836,
    0.408282678752175,0.591717321247825,0.762766204958164,0.898333238706813,0.980144928248768);
const float GW[8] = float[8](0.050614268145188,0.111190517226687,0.156853322938944,
    0.181341891689181,0.181341891689181,0.156853322938944,0.111190517226687,0.050614268145188);
void main(){
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(N*N)) return;
    int m = int(idx)/N, n = int(idx)%N;
    vec2 s1 = vec2(0.0), s2 = vec2(0.0);
    for (int pi = 0; pi < 2; ++pi) {
        vec4 PA = bg[m*4 + pi*2], PB = bg[m*4 + pi*2 + 1];
        vec3 Pa = PA.xyz, Pb = PB.xyz;
        float Plen = length(Pb - Pa);
        vec3 Pt = (Pb - Pa) / Plen;
        float Psgn = PA.w; float Pramp = PB.w;
        for (int qi = 0; qi < 2; ++qi) {
            vec4 QA = bg[n*4 + qi*2], QB = bg[n*4 + qi*2 + 1];
            vec3 Qa = QA.xyz, Qb = QB.xyz;
            float Qlen = length(Qb - Qa);
            vec3 Qt = (Qb - Qa) / Qlen;
            float Qsgn = QA.w; float Qramp = QB.w;
            float tdot = dot(Pt, Qt);
            vec2 i1 = vec2(0.0), i0 = vec2(0.0);
            for (int g = 0; g < 8; ++g) {
                float ug = GU[g];
                vec3 rp = Pa + ug*(Pb - Pa);
                float rampm = (Pramp < 0.5) ? ug : (1.0 - ug);
                for (int h = 0; h < 8; ++h) {
                    float uh = GU[h];
                    vec3 rq = Qa + uh*(Qb - Qa);
                    float rampn = (Qramp < 0.5) ? uh : (1.0 - uh);
                    vec3 dd = rp - rq;
                    float R = sqrt(dot(dd,dd) + a2);
                    float ww = GW[g]*GW[h];
                    vec2 G = vec2(cos(kk*R), -sin(kk*R)) / (4.0*3.14159265358979*R);
                    i1 += (ww*rampm*rampn) * G;
                    i0 += ww * G;
                }
            }
            s1 += (tdot*Plen*Qlen) * i1;
            s2 += (Psgn*Qsgn) * i0;
        }
    }
    // Zmn = j*wmu*s1 - j*(wmu/kk^2)*s2     (multiply by j = rotate)
    vec2 t1 = wmu * vec2(-s1.y, s1.x);
    float f2 = wmu/(kk*kk);
    vec2 t2 = f2 * vec2(-s2.y, s2.x);
    Z[idx] = t1 - t2;
}
)";

static const char *MOM_DIAG = R"(
layout(std430, binding=1) buffer BZ { vec2 Z[]; };
layout(std430, binding=2) buffer BD { vec2 dg[]; };
uniform int N;
void main(){
    uint m = gl_GlobalInvocationID.x;
    if (m >= uint(N)) return;
    dg[m] = Z[m*uint(N) + m];
}
)";

static const char *MOM_SPMV = R"(
layout(std430, binding=1) buffer BZ { vec2 Z[]; };
layout(std430, binding=6) buffer BP { vec2 pv[]; };
layout(std430, binding=7) buffer BQ { vec2 qv[]; };
uniform int N;
void main(){
    uint m = gl_GlobalInvocationID.x;
    if (m >= uint(N)) return;
    vec2 s = vec2(0.0);
    uint base = m*uint(N);
    for (int n = 0; n < N; ++n)
        s += cmul(Z[base + uint(n)], pv[n]);
    qv[m] = s;
}
)";

static const char *MOM_PRE = R"(
layout(std430, binding=2) buffer BD { vec2 dg[]; };
layout(std430, binding=4) buffer BR { vec2 rv[]; };
layout(std430, binding=5) buffer BZv { vec2 zv[]; };
uniform int N;
void main(){ uint i=gl_GlobalInvocationID.x; if(i>=uint(N)) return; zv[i]=cdiv(rv[i],dg[i]); }
)";

static const char *MOM_XR = R"(
layout(std430, binding=3) buffer BX { vec2 xv[]; };
layout(std430, binding=4) buffer BR { vec2 rv[]; };
layout(std430, binding=6) buffer BP { vec2 pv[]; };
layout(std430, binding=7) buffer BQ { vec2 qv[]; };
uniform int N; uniform float aRe, aIm;
void main(){ uint i=gl_GlobalInvocationID.x; if(i>=uint(N)) return;
    vec2 a=vec2(aRe,aIm); xv[i]+=cmul(a,pv[i]); rv[i]-=cmul(a,qv[i]); }
)";

static const char *MOM_PUPD = R"(
layout(std430, binding=5) buffer BZv { vec2 zv[]; };
layout(std430, binding=6) buffer BP  { vec2 pv[]; };
uniform int N; uniform float bRe, bIm;
void main(){ uint i=gl_GlobalInvocationID.x; if(i>=uint(N)) return;
    vec2 b=vec2(bRe,bIm); pv[i]=zv[i]+cmul(b,pv[i]); }
)";

static const char *MOM_DOT = R"(
layout(std430, binding=4) buffer BR { vec2 rv[]; };
layout(std430, binding=5) buffer BZv{ vec2 zv[]; };
layout(std430, binding=6) buffer BP { vec2 pv[]; };
layout(std430, binding=7) buffer BQ { vec2 qv[]; };
layout(std430, binding=8) buffer BPC{ vec2 part[]; };
uniform int N, np, which;
void main(){
    uint tid=gl_GlobalInvocationID.x; if(tid>=uint(np)) return;
    vec2 s=vec2(0.0);
    for (uint i=tid; i<uint(N); i+=uint(np)) {
        vec2 a=(which==0)?pv[i]:rv[i];
        vec2 b=(which==0)?qv[i]:zv[i];
        s += cmul(a,b);
    }
    part[tid]=s;
}
)";

static const char *MOM_NRM = R"(
layout(std430, binding=4)  buffer BR { vec2 rv[]; };
layout(std430, binding=9)  buffer BPR{ float pr[]; };
uniform int N, np;
void main(){
    uint tid=gl_GlobalInvocationID.x; if(tid>=uint(np)) return;
    float s=0.0;
    for (uint i=tid; i<uint(N); i+=uint(np)) { vec2 v=rv[i]; s+=v.x*v.x+v.y*v.y; }
    pr[tid]=s;
}
)";

//---------------------------------------------------------------------------
bool RunGpuMom(MomSolver &s, std::string &msg)
{
    typedef std::complex<double> cplx;
    const int N = s.N;
    if (N <= 0) { msg = "no unknowns"; return false; }

    GpuCompute gc;
    if (!gc.ok()) { msg = gc.error(); return false; }

    auto prog = [&](const char *body) -> unsigned
    {
        std::string src = std::string(MOM_COMMON) + body;
        return gc.buildProgram(src.c_str());
    };
    unsigned pFill = prog(MOM_FILL); if (!pFill) { msg = gc.error(); return false; }
    unsigned pDiag = prog(MOM_DIAG); if (!pDiag) { msg = gc.error(); return false; }
    unsigned pSp   = prog(MOM_SPMV); if (!pSp)   { msg = gc.error(); return false; }
    unsigned pPre  = prog(MOM_PRE);  if (!pPre)  { msg = gc.error(); return false; }
    unsigned pXr   = prog(MOM_XR);   if (!pXr)   { msg = gc.error(); return false; }
    unsigned pPu   = prog(MOM_PUPD); if (!pPu)   { msg = gc.error(); return false; }
    unsigned pDot  = prog(MOM_DOT);  if (!pDot)  { msg = gc.error(); return false; }
    unsigned pNrm  = prog(MOM_NRM);  if (!pNrm)  { msg = gc.error(); return false; }

    // ---- pack basis geometry: 4 vec4 per basis (A0,B0,A1,B1) ----
    std::vector<float> bg((size_t)N * 16, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < 2; ++k)
        {
            const MomSolver::SubSeg &S = s.basis[i].sub[k];
            float sgn = S.dsign * S.len;       // +-1
            float *A = &bg[(size_t)i*16 + k*8];
            A[0]=S.a.x; A[1]=S.a.y; A[2]=S.a.z; A[3]=sgn;
            A[4]=S.b.x; A[5]=S.b.y; A[6]=S.b.z; A[7]=(float)S.rampType;
        }

    std::vector<float> rhs((size_t)N*2, 0.0f);
    for (int i = 0; i < N; ++i) { rhs[i*2]=(float)s.V[i].real(); rhs[i*2+1]=(float)s.V[i].imag(); }

    const int NP = 256;
    gc.glErr();
    unsigned bBg = gc.makeBuffer(0, bg.size()*4, bg.data());
    unsigned bZ  = gc.makeBuffer(1, (size_t)N*N*2*sizeof(float), nullptr);
    unsigned bD  = gc.makeBuffer(2, (size_t)N*2*sizeof(float), nullptr);
    unsigned bX  = gc.makeBuffer(3, (size_t)N*2*sizeof(float), nullptr);
    unsigned bR  = gc.makeBuffer(4, (size_t)N*2*sizeof(float), rhs.data());
    unsigned bZv = gc.makeBuffer(5, (size_t)N*2*sizeof(float), nullptr);
    unsigned bP  = gc.makeBuffer(6, (size_t)N*2*sizeof(float), nullptr);
    unsigned bQ  = gc.makeBuffer(7, (size_t)N*2*sizeof(float), nullptr);
    unsigned bPc = gc.makeBuffer(8, (size_t)NP*2*sizeof(float), nullptr);
    unsigned bPr = gc.makeBuffer(9, (size_t)NP*sizeof(float), nullptr);
    (void)bBg; (void)bD; (void)bX; (void)bZv; (void)bQ;
    if (gc.glErr() != 0) { msg = "GPU buffer allocation failed"; return false; }

    // ---- fill matrix ----
    const double w = 2.0 * M_PI * s.f0;
    const double k = w / C0;
    gc.use(pFill);
    gc.set1i(gc.uniform(pFill, "N"), N);
    gc.set1f(gc.uniform(pFill, "kk"), (float)k);
    gc.set1f(gc.uniform(pFill, "wmu"), (float)(w * MU0));
    gc.set1f(gc.uniform(pFill, "a2"), (float)((double)s.radius * s.radius));
    gc.dispatch((size_t)N * N);
    gc.barrier();
    gc.use(pDiag); gc.set1i(gc.uniform(pDiag, "N"), N); gc.dispatch(N); gc.barrier();

    auto setN = [&](unsigned pr){ gc.use(pr); gc.set1i(gc.uniform(pr, "N"), N); };
    setN(pSp); setN(pPre); setN(pXr); setN(pPu);
    gc.use(pDot); gc.set1i(gc.uniform(pDot,"N"),N); gc.set1i(gc.uniform(pDot,"np"),NP);
    gc.use(pNrm); gc.set1i(gc.uniform(pNrm,"N"),N); gc.set1i(gc.uniform(pNrm,"np"),NP);

    std::vector<float> partC(NP*2), partR(NP);
    auto cdot = [&](int which) -> cplx
    {
        gc.use(pDot); gc.set1i(gc.uniform(pDot,"which"), which);
        gc.dispatchGroups((NP+63)/64); gc.barrier();
        gc.readBuffer(bPc, (size_t)NP*2*sizeof(float), partC.data());
        double re=0, im=0;
        for (int i=0;i<NP;++i){ re+=partC[i*2]; im+=partC[i*2+1]; }
        return cplx(re, im);
    };
    auto rn2 = [&]() -> double
    {
        gc.use(pNrm); gc.dispatchGroups((NP+63)/64); gc.barrier();
        gc.readBuffer(bPr, (size_t)NP*sizeof(float), partR.data());
        double t=0; for(int i=0;i<NP;++i) t+=partR[i]; return t;
    };

    double bn = std::sqrt(rn2());        // r = V at start
    if (bn < 1e-30) { msg = "zero excitation"; return false; }
    gc.use(pPre); gc.dispatch(N); gc.barrier();         // z = r/diag
    gc.use(pPu);                                          // p = z (+0*p)
    gc.set1f(gc.uniform(pPu,"bRe"),0.0f); gc.set1f(gc.uniform(pPu,"bIm"),0.0f);
    gc.dispatch(N); gc.barrier();
    cplx rho = cdot(1);
    const double tol = 1e-5;
    const int maxIt = std::max(200, 4*N);

    int it = 0;
    for (; it < maxIt && !s.stopFlag; ++it)
    {
        gc.use(pSp); gc.dispatch(N); gc.barrier();      // q = Z p
        cplx pq = cdot(0);
        if (std::abs(pq) < 1e-30) break;
        cplx alpha = rho / pq;
        gc.use(pXr);
        gc.set1f(gc.uniform(pXr,"aRe"),(float)alpha.real());
        gc.set1f(gc.uniform(pXr,"aIm"),(float)alpha.imag());
        gc.dispatch(N); gc.barrier();
        double res = std::sqrt(rn2()) / bn;
        s.resNorm = (float)res;
        s.curStep = it + 1;
        if (res < tol) { ++it; break; }
        gc.use(pPre); gc.dispatch(N); gc.barrier();
        cplx rho1 = cdot(1);
        if (std::abs(rho) < 1e-30) break;
        cplx beta = rho1 / rho; rho = rho1;
        gc.use(pPu);
        gc.set1f(gc.uniform(pPu,"bRe"),(float)beta.real());
        gc.set1f(gc.uniform(pPu,"bIm"),(float)beta.imag());
        gc.dispatch(N); gc.barrier();
    }

    std::vector<float> xf((size_t)N*2);
    gc.readBuffer(bX, (size_t)N*2*sizeof(float), xf.data());
    s.I.assign(N, cplx(0,0));
    for (int i = 0; i < N; ++i)
        s.I[i] = cplx(xf[i*2], xf[i*2+1]);

    msg = gc.rendererName();
    return true;
}

//---------------------------------------------------------------------------
// surface RWG fill kernel: one thread per matrix entry (m,n)
//---------------------------------------------------------------------------
static const char *MOMS_FILL = R"(
layout(std430, binding=1)  buffer BZ  { vec2 Z[]; };
layout(std430, binding=10) buffer BTD { vec4 td[]; };   // 4 vec4 / triangle
layout(std430, binding=11) buffer BRT { ivec4 rt[]; };  // triP,triM per rwg
layout(std430, binding=12) buffer BRG { vec4 rg[]; };   // vP(xyz,len), vM(xyz,0)
uniform int N;
uniform float kk, c1, c2, wom;
const float TA[7]=float[7](0.333333333,0.059715872,0.470142064,0.470142064,0.797426985,0.101286507,0.101286507);
const float TB[7]=float[7](0.333333333,0.470142064,0.059715872,0.470142064,0.101286507,0.797426985,0.101286507);
const float TW[7]=float[7](0.225,0.132394153,0.132394153,0.132394153,0.125939181,0.125939181,0.125939181);
const float DG[4]=float[4](0.069431844,0.330009478,0.669990522,0.930568156);
const float DW[4]=float[4](0.173927423,0.326072577,0.326072577,0.173927423);

void triInt(vec3 fp, int t, bool self, out vec2 Ig, out vec2 Ix, out vec2 Iy, out vec2 Iz){
    Ig=vec2(0); Ix=vec2(0); Iy=vec2(0); Iz=vec2(0);
    vec3 v0=td[t*4].xyz, v1=td[t*4+1].xyz, v2=td[t*4+2].xyz;
    float A=td[t*4+3].w;
    if (!self) {
        for (int q=0;q<7;++q){
            float a=TA[q], b=TB[q], c=1.0-a-b;
            vec3 rp=v0*a+v1*b+v2*c;
            float R=length(fp-rp); if(R<1e-12) continue;
            vec2 G=vec2(cos(kk*R),-sin(kk*R))/R; float wA=TW[q]*A;
            Ig+=wA*G; Ix+=wA*rp.x*G; Iy+=wA*rp.y*G; Iz+=wA*rp.z*G;
        }
    } else {
        vec3 C=(v0+v1+v2)/3.0;
        vec3 cor[3]=vec3[3](v0,v1,v2);
        for (int s=0;s<3;++s){
            vec3 B=cor[s], D=cor[(s+1)%3];
            float subA=0.5*length(cross(B-C,D-C));
            for(int iu=0;iu<4;++iu){ float u=DG[iu];
                for(int iv=0;iv<4;++iv){ float vv=DG[iv];
                    vec3 rp=C+((B-C)*(1.0-vv)+(D-C)*vv)*u;
                    float R=length(fp-rp); if(R<1e-12) continue;
                    vec2 G=vec2(cos(kk*R),-sin(kk*R))/R;
                    float wA=DW[iu]*DW[iv]*2.0*subA*u;
                    Ig+=wA*G; Ix+=wA*rp.x*G; Iy+=wA*rp.y*G; Iz+=wA*rp.z*G;
                }}
        }
    }
}
void main(){
    uint idx=gl_GlobalInvocationID.x; if(idx>=uint(N*N)) return;
    int m=int(idx)/N, n=int(idx)%N;
    int mt[2]=int[2](rt[m].x, rt[m].y);
    vec3 mFree[2]=vec3[2](rg[m*2].xyz, rg[m*2+1].xyz);
    float lenM=rg[m*2].w;
    int nt[2]=int[2](rt[n].x, rt[n].y);
    vec3 nFree[2]=vec3[2](rg[n*2].xyz, rg[n*2+1].xyz);
    float lenN=rg[n*2].w;
    float mSgn[2]=float[2](1.0,-1.0), nSgn[2]=float[2](1.0,-1.0);
    vec2 Zmn=vec2(0);
    for (int sp=0; sp<2; ++sp){
        vec3 rc=td[mt[sp]*4+3].xyz;          // centroid stored in td[*4+3].xyz
        vec3 rhoM=(mFree[sp]-rc)*(-mSgn[sp]);
        vec2 Ax=vec2(0),Ay=vec2(0),Az=vec2(0),Phi=vec2(0);
        for (int sq=0; sq<2; ++sq){
            bool self=(mt[sp]==nt[sq]);
            float An=td[nt[sq]*4+3].w;
            vec2 Ig,Ix,Iy,Iz; triInt(rc, nt[sq], self, Ig,Ix,Iy,Iz);
            float cf=lenN/(2.0*An)*nSgn[sq];
            Ax += (c1*cf)*(Ix - nFree[sq].x*Ig);
            Ay += (c1*cf)*(Iy - nFree[sq].y*Ig);
            Az += (c1*cf)*(Iz - nFree[sq].z*Ig);
            float dn=nSgn[sq]*lenN/An;
            Phi += (c2*dn)*vec2(-Ig.y, Ig.x);     // j*Ig
        }
        vec2 AdotRho = Ax*rhoM.x + Ay*rhoM.y + Az*rhoM.z;
        vec2 jwAR = wom*vec2(-AdotRho.y, AdotRho.x);  // jw*(A.rho)
        Zmn += lenM*( 0.5*jwAR + (sp==1 ? Phi : -Phi) );
    }
    Z[idx]=Zmn;
}
)";

bool RunGpuMomSurf(MomSurface &s, std::string &msg)
{
    typedef std::complex<double> cplx;
    const int N = s.N;
    if (N <= 0) { msg = "no unknowns"; return false; }

    GpuCompute gc;
    if (!gc.ok()) { msg = gc.error(); return false; }
    auto prog = [&](const char *body) -> unsigned
    { std::string src = std::string(MOM_COMMON) + body; return gc.buildProgram(src.c_str()); };
    unsigned pFill=prog(MOMS_FILL); if(!pFill){msg=gc.error();return false;}
    unsigned pDiag=prog(MOM_DIAG);  if(!pDiag){msg=gc.error();return false;}
    unsigned pSp  =prog(MOM_SPMV);  if(!pSp){msg=gc.error();return false;}
    unsigned pPre =prog(MOM_PRE);   if(!pPre){msg=gc.error();return false;}
    unsigned pXr  =prog(MOM_XR);    if(!pXr){msg=gc.error();return false;}
    unsigned pPu  =prog(MOM_PUPD);  if(!pPu){msg=gc.error();return false;}
    unsigned pDot =prog(MOM_DOT);   if(!pDot){msg=gc.error();return false;}
    unsigned pNrm =prog(MOM_NRM);   if(!pNrm){msg=gc.error();return false;}

    // triangle data: 4 vec4 / tri (v0,v1,v2, centroid.xyz+area.w)
    int nt = (int)s.triA.size();
    std::vector<float> td((size_t)nt*16, 0.0f);
    for (int t = 0; t < nt; ++t)
    {
        for (int c = 0; c < 3; ++c)
        {
            int vi = s.tris[t*3+c];
            td[t*16 + c*4 + 0] = s.verts[vi*3];
            td[t*16 + c*4 + 1] = s.verts[vi*3+1];
            td[t*16 + c*4 + 2] = s.verts[vi*3+2];
        }
        td[t*16 + 12] = s.triC[t].x;
        td[t*16 + 13] = s.triC[t].y;
        td[t*16 + 14] = s.triC[t].z;
        td[t*16 + 15] = s.triA[t];
    }
    std::vector<int>   rt((size_t)N*4, 0);
    std::vector<float> rg((size_t)N*8, 0.0f);
    for (int m = 0; m < N; ++m)
    {
        rt[m*4]   = s.rwg[m].triP;
        rt[m*4+1] = s.rwg[m].triM;
        rg[m*8]   = s.rwg[m].vP.x; rg[m*8+1]=s.rwg[m].vP.y; rg[m*8+2]=s.rwg[m].vP.z;
        rg[m*8+3] = s.rwg[m].len;
        rg[m*8+4] = s.rwg[m].vM.x; rg[m*8+5]=s.rwg[m].vM.y; rg[m*8+6]=s.rwg[m].vM.z;
    }
    std::vector<float> rhs((size_t)N*2, 0.0f);
    for (int i = 0; i < N; ++i){ rhs[i*2]=(float)s.V[i].real(); rhs[i*2+1]=(float)s.V[i].imag(); }

    const int NP = 256;
    gc.glErr();
    unsigned bZ =gc.makeBuffer(1,(size_t)N*N*2*sizeof(float),nullptr);
    unsigned bD =gc.makeBuffer(2,(size_t)N*2*sizeof(float),nullptr);
    unsigned bX =gc.makeBuffer(3,(size_t)N*2*sizeof(float),nullptr);
    unsigned bR =gc.makeBuffer(4,(size_t)N*2*sizeof(float),rhs.data());
    unsigned bZv=gc.makeBuffer(5,(size_t)N*2*sizeof(float),nullptr);
    unsigned bP =gc.makeBuffer(6,(size_t)N*2*sizeof(float),nullptr);
    unsigned bQ =gc.makeBuffer(7,(size_t)N*2*sizeof(float),nullptr);
    unsigned bPc=gc.makeBuffer(8,(size_t)NP*2*sizeof(float),nullptr);
    unsigned bPr=gc.makeBuffer(9,(size_t)NP*sizeof(float),nullptr);
    gc.makeBuffer(10, td.size()*4, td.data());
    gc.makeBuffer(11, rt.size()*4, rt.data());
    gc.makeBuffer(12, rg.size()*4, rg.data());
    (void)bD;(void)bX;(void)bZv;(void)bQ;
    if (gc.glErr()!=0){ msg="GPU buffer allocation failed"; return false; }

    const double w = 2.0*M_PI*s.f0, k = w/C0;
    const double fourPi = 4.0*M_PI;
    gc.use(pFill);
    gc.set1i(gc.uniform(pFill,"N"),N);
    gc.set1f(gc.uniform(pFill,"kk"),(float)k);
    gc.set1f(gc.uniform(pFill,"c1"),(float)(MU0/fourPi));
    gc.set1f(gc.uniform(pFill,"c2"),(float)(1.0/(fourPi*w*8.8541878128e-12)));
    gc.set1f(gc.uniform(pFill,"wom"),(float)w);
    gc.dispatch((size_t)N*N); gc.barrier();
    gc.use(pDiag); gc.set1i(gc.uniform(pDiag,"N"),N); gc.dispatch(N); gc.barrier();

    auto setN=[&](unsigned pr){ gc.use(pr); gc.set1i(gc.uniform(pr,"N"),N); };
    setN(pSp); setN(pPre); setN(pXr); setN(pPu);
    gc.use(pDot); gc.set1i(gc.uniform(pDot,"N"),N); gc.set1i(gc.uniform(pDot,"np"),NP);
    gc.use(pNrm); gc.set1i(gc.uniform(pNrm,"N"),N); gc.set1i(gc.uniform(pNrm,"np"),NP);

    std::vector<float> partC(NP*2), partR(NP);
    auto cdot=[&](int which)->cplx{
        gc.use(pDot); gc.set1i(gc.uniform(pDot,"which"),which);
        gc.dispatchGroups((NP+63)/64); gc.barrier();
        gc.readBuffer(bPc,(size_t)NP*2*sizeof(float),partC.data());
        double re=0,im=0; for(int i=0;i<NP;++i){re+=partC[i*2];im+=partC[i*2+1];}
        return cplx(re,im); };
    auto rn2=[&]()->double{
        gc.use(pNrm); gc.dispatchGroups((NP+63)/64); gc.barrier();
        gc.readBuffer(bPr,(size_t)NP*sizeof(float),partR.data());
        double t=0; for(int i=0;i<NP;++i)t+=partR[i]; return t; };

    double bn=std::sqrt(rn2()); if(bn<1e-30){msg="zero excitation";return false;}
    gc.use(pPre); gc.dispatch(N); gc.barrier();
    gc.use(pPu); gc.set1f(gc.uniform(pPu,"bRe"),0.0f); gc.set1f(gc.uniform(pPu,"bIm"),0.0f);
    gc.dispatch(N); gc.barrier();
    cplx rho=cdot(1); const double tol=1e-5; const int maxIt=std::max(400,6*N);
    int it=0;
    for(; it<maxIt && !s.stopFlag; ++it){
        gc.use(pSp); gc.dispatch(N); gc.barrier();
        cplx pq=cdot(0); if(std::abs(pq)<1e-30) break;
        cplx alpha=rho/pq;
        gc.use(pXr); gc.set1f(gc.uniform(pXr,"aRe"),(float)alpha.real());
        gc.set1f(gc.uniform(pXr,"aIm"),(float)alpha.imag());
        gc.dispatch(N); gc.barrier();
        double res=std::sqrt(rn2())/bn; s.resNorm=(float)res; s.curStep=it+1;
        if(res<tol){++it;break;}
        gc.use(pPre); gc.dispatch(N); gc.barrier();
        cplx rho1=cdot(1); if(std::abs(rho)<1e-30) break;
        cplx beta=rho1/rho; rho=rho1;
        gc.use(pPu); gc.set1f(gc.uniform(pPu,"bRe"),(float)beta.real());
        gc.set1f(gc.uniform(pPu,"bIm"),(float)beta.imag());
        gc.dispatch(N); gc.barrier();
    }
    double finalRes = s.resNorm;
    if (finalRes < 1e-2)
    {
        std::vector<float> xf((size_t)N*2);
        gc.readBuffer(bX,(size_t)N*2*sizeof(float),xf.data());
        s.J.assign(N, cplx(0,0));
        for(int i=0;i<N;++i) s.J[i]=cplx(xf[i*2], xf[i*2+1]);
        msg = gc.rendererName() + " (COCG)";
        return true;
    }

    // COCG stalled (EFIE surface matrices are ill-conditioned): read the
    // GPU-filled matrix back and solve directly on the CPU. The expensive
    // O(N^2 * quadrature) fill still ran on the GPU.
    std::vector<float> zf((size_t)N*N*2);
    gc.readBuffer(bZ, zf.size()*sizeof(float), zf.data());
    std::vector<cplx> A((size_t)N*N);
    for (size_t i = 0; i < (size_t)N*N; ++i)
        A[i] = cplx(zf[i*2], zf[i*2+1]);
    std::vector<cplx> x = s.V;
    for (int col = 0; col < N; ++col)
    {
        int piv = col; double best = std::abs(A[(size_t)col*N+col]);
        for (int r = col+1; r < N; ++r)
        { double v=std::abs(A[(size_t)r*N+col]); if(v>best){best=v;piv=r;} }
        if (piv != col)
        {
            for (int j = 0; j < N; ++j)
                std::swap(A[(size_t)col*N+j], A[(size_t)piv*N+j]);
            std::swap(x[col], x[piv]);
        }
        cplx d = A[(size_t)col*N+col];
        if (std::abs(d) < 1e-300) continue;
        for (int r = col+1; r < N; ++r)
        {
            cplx fct = A[(size_t)r*N+col]/d;
            for (int j = col; j < N; ++j)
                A[(size_t)r*N+j] -= fct*A[(size_t)col*N+j];
            x[r] -= fct*x[col];
        }
    }
    for (int r = N-1; r >= 0; --r)
    {
        cplx ssum = x[r];
        for (int j = r+1; j < N; ++j) ssum -= A[(size_t)r*N+j]*x[j];
        cplx d = A[(size_t)r*N+r];
        x[r] = (std::abs(d)>1e-300) ? ssum/d : cplx(0,0);
    }
    s.J = x;
    s.resNorm = 0.0f;
    msg = gc.rendererName() + " (GPU fill + direct)";
    return true;
}
