//---------------------------------------------------------------------------
// GpuFdtd.cpp - Yee FDTD time loop on the GPU via OpenGL 4.3 compute shaders
//
// Mirrors FdtdSolver's CPU algorithm (FdtdSolver.cpp): six staggered field
// SSBOs, per-edge update coefficients (vec2 = ca,cb; PEC -> 0), precomputed
// soft-source coefficients, first-order Mur boundaries (via a one-step copy
// of E into prevE), and the same monitors (Js DFT, Huygens DFT, port V/I,
// energy, cut plane). Results are filled into the solver's member arrays so
// finalizeDft() and the UI work identically to the CPU path. Any failure
// returns false -> CPU fallback.
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GpuFdtd.h"
#include "FdtdSolver.h"
#include "GpuCompute.h"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#pragma package(smart_init)

static const float ETA0 = 376.730313f;
static const float MU0  = 1.25663706212e-6f;

//---------------------------------------------------------------------------
// GLSL building blocks
//---------------------------------------------------------------------------
static const char *FD_HDR = R"(
#version 430
layout(local_size_x = 64) in;
uniform ivec3 dims;            // nx, ny, nz
int IEX(int i,int j,int k){ return (k*(dims.y+1)+j)*dims.x + i; }
int IEY(int i,int j,int k){ return (k*dims.y+j)*(dims.x+1) + i; }
int IEZ(int i,int j,int k){ return (k*(dims.y+1)+j)*(dims.x+1) + i; }
int IHX(int i,int j,int k){ return (k*dims.y+j)*(dims.x+1) + i; }
int IHY(int i,int j,int k){ return (k*(dims.y+1)+j)*dims.x + i; }
int IHZ(int i,int j,int k){ return (k*dims.y+j)*dims.x + i; }
)";

// field-buffer declarations + cell-centered averaging (for monitor kernels)
static const char *FD_FIELDS = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=3) buffer BHx { float Hx[]; };
layout(std430, binding=4) buffer BHy { float Hy[]; };
layout(std430, binding=5) buffer BHz { float Hz[]; };
float ecX(int i,int j,int k){return 0.25*(Ex[IEX(i,j,k)]+Ex[IEX(i,j+1,k)]+Ex[IEX(i,j,k+1)]+Ex[IEX(i,j+1,k+1)]);}
float ecY(int i,int j,int k){return 0.25*(Ey[IEY(i,j,k)]+Ey[IEY(i+1,j,k)]+Ey[IEY(i,j,k+1)]+Ey[IEY(i+1,j,k+1)]);}
float ecZ(int i,int j,int k){return 0.25*(Ez[IEZ(i,j,k)]+Ez[IEZ(i+1,j,k)]+Ez[IEZ(i,j+1,k)]+Ez[IEZ(i+1,j+1,k)]);}
float hcX(int i,int j,int k){return 0.5*(Hx[IHX(i,j,k)]+Hx[IHX(i+1,j,k)]);}
float hcY(int i,int j,int k){return 0.5*(Hy[IHY(i,j,k)]+Hy[IHY(i,j+1,k)]);}
float hcZ(int i,int j,int k){return 0.5*(Hz[IHZ(i,j,k)]+Hz[IHZ(i,j,k+1)]);}
float eAt(int i,int j,int k,int c){return c==0?ecX(i,j,k):(c==1?ecY(i,j,k):ecZ(i,j,k));}
float hAt(int i,int j,int k,int c){return c==0?hcX(i,j,k):(c==1?hcY(i,j,k):hcZ(i,j,k));}
const ivec2 POLT[3] = ivec2[3](ivec2(1,2), ivec2(0,2), ivec2(0,1));
const ivec2 CYC[3]  = ivec2[3](ivec2(1,2), ivec2(2,0), ivec2(0,1));
void cellCo(int cell, out int i, out int j, out int k){
    i = cell % dims.x; j = (cell/dims.x) % dims.y; k = cell/(dims.x*dims.y);
}
)";

// inject soft sources: E += src * sv  (per-edge coefficient, race free)
static const char *FD_INJECT = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=9)  buffer BSx { float Sx[]; };
layout(std430, binding=10) buffer BSy { float Sy[]; };
layout(std430, binding=11) buffer BSz { float Sz[]; };
uniform int nEx, nEy, nEz;
uniform float sv;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g < uint(nEx)) Ex[g] += Sx[g] * sv;
    if (g < uint(nEy)) Ey[g] += Sy[g] * sv;
    if (g < uint(nEz)) Ez[g] += Sz[g] * sv;
}
)";

// H update (all three components)
static const char *FD_UPDH = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=3) buffer BHx { float Hx[]; };
layout(std430, binding=4) buffer BHy { float Hy[]; };
layout(std430, binding=5) buffer BHz { float Hz[]; };
uniform int nHx, nHy, nHz;
uniform float ch;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g < uint(nHx)) {
        int i = int(g) % (dims.x+1);
        int j = (int(g)/(dims.x+1)) % dims.y;
        int k = int(g)/((dims.x+1)*dims.y);
        Hx[g] -= ch * ((Ez[IEZ(i,j+1,k)]-Ez[IEZ(i,j,k)]) -
                       (Ey[IEY(i,j,k+1)]-Ey[IEY(i,j,k)]));
    }
    if (g < uint(nHy)) {
        int i = int(g) % dims.x;
        int j = (int(g)/dims.x) % (dims.y+1);
        int k = int(g)/(dims.x*(dims.y+1));
        Hy[g] -= ch * ((Ex[IEX(i,j,k+1)]-Ex[IEX(i,j,k)]) -
                       (Ez[IEZ(i+1,j,k)]-Ez[IEZ(i,j,k)]));
    }
    if (g < uint(nHz)) {
        int i = int(g) % dims.x;
        int j = (int(g)/dims.x) % dims.y;
        int k = int(g)/(dims.x*dims.y);
        Hz[g] -= ch * ((Ey[IEY(i+1,j,k)]-Ey[IEY(i,j,k)]) -
                       (Ex[IEX(i,j+1,k)]-Ex[IEX(i,j,k)]));
    }
}
)";

// copy E -> prevE (snapshot for Mur)
static const char *FD_COPY = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=12) buffer BPx { float Px[]; };
layout(std430, binding=13) buffer BPy { float Py[]; };
layout(std430, binding=14) buffer BPz { float Pz[]; };
uniform int nEx, nEy, nEz;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g < uint(nEx)) Px[g] = Ex[g];
    if (g < uint(nEy)) Py[g] = Ey[g];
    if (g < uint(nEz)) Pz[g] = Ez[g];
}
)";

// E update (interior nodes only; PEC edges -> 0). cE = vec2(ca, cb).
static const char *FD_UPDE = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=3) buffer BHx { float Hx[]; };
layout(std430, binding=4) buffer BHy { float Hy[]; };
layout(std430, binding=5) buffer BHz { float Hz[]; };
layout(std430, binding=6) buffer BCx { vec2 cEx[]; };
layout(std430, binding=7) buffer BCy { vec2 cEy[]; };
layout(std430, binding=8) buffer BCz { vec2 cEz[]; };
uniform int nEx, nEy, nEz;
void main(){
    uint g = gl_GlobalInvocationID.x;
    int nx = dims.x, ny = dims.y, nz = dims.z;
    if (g < uint(nEx)) {
        int i = int(g) % nx;
        int j = (int(g)/nx) % (ny+1);
        int k = int(g)/(nx*(ny+1));
        if (j>=1 && j<ny && k>=1 && k<nz) {
            vec2 c = cEx[g];
            float curl = (Hz[IHZ(i,j,k)]-Hz[IHZ(i,j-1,k)]) -
                         (Hy[IHY(i,j,k)]-Hy[IHY(i,j,k-1)]);
            Ex[g] = c.x*Ex[g] + c.y*curl;   // PEC: c=(0,0) -> 0
        }
    }
    if (g < uint(nEy)) {
        int i = int(g) % (nx+1);
        int j = (int(g)/(nx+1)) % ny;
        int k = int(g)/((nx+1)*ny);
        if (i>=1 && i<nx && k>=1 && k<nz) {
            vec2 c = cEy[g];
            float curl = (Hx[IHX(i,j,k)]-Hx[IHX(i,j,k-1)]) -
                         (Hz[IHZ(i,j,k)]-Hz[IHZ(i-1,j,k)]);
            Ey[g] = c.x*Ey[g] + c.y*curl;
        }
    }
    if (g < uint(nEz)) {
        int i = int(g) % (nx+1);
        int j = (int(g)/(nx+1)) % (ny+1);
        int k = int(g)/((nx+1)*(ny+1));
        if (i>=1 && i<nx && j>=1 && j<ny) {
            vec2 c = cEz[g];
            float curl = (Hy[IHY(i,j,k)]-Hy[IHY(i-1,j,k)]) -
                         (Hx[IHX(i,j,k)]-Hx[IHX(i,j-1,k)]);
            Ez[g] = c.x*Ez[g] + c.y*curl;
        }
    }
}
)";

// first-order Mur on the 6 outer faces using prevE snapshots
static const char *FD_MUR = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=12) buffer BPx { float Px[]; };
layout(std430, binding=13) buffer BPy { float Py[]; };
layout(std430, binding=14) buffer BPz { float Pz[]; };
uniform int nEx, nEy, nEz;
uniform float cMur;
void main(){
    uint g = gl_GlobalInvocationID.x;
    int nx = dims.x, ny = dims.y, nz = dims.z;
    // Ex tangential on y and z faces
    if (g < uint(nEx)) {
        int i = int(g) % nx;
        int j = (int(g)/nx) % (ny+1);
        int k = int(g)/(nx*(ny+1));
        int eb=-1, e1=-1;
        if (j==0)       { eb=IEX(i,0,k);     e1=IEX(i,1,k); }
        else if (j==ny) { eb=IEX(i,ny,k);    e1=IEX(i,ny-1,k); }
        else if (k==0)  { eb=IEX(i,j,0);     e1=IEX(i,j,1); }
        else if (k==nz) { eb=IEX(i,j,nz);    e1=IEX(i,j,nz-1); }
        if (eb>=0) Ex[eb] = Px[e1] + cMur*(Ex[e1]-Px[eb]);
    }
    // Ey tangential on x and z faces
    if (g < uint(nEy)) {
        int i = int(g) % (nx+1);
        int j = (int(g)/(nx+1)) % ny;
        int k = int(g)/((nx+1)*ny);
        int eb=-1, e1=-1;
        if (i==0)       { eb=IEY(0,j,k);     e1=IEY(1,j,k); }
        else if (i==nx) { eb=IEY(nx,j,k);    e1=IEY(nx-1,j,k); }
        else if (k==0)  { eb=IEY(i,j,0);     e1=IEY(i,j,1); }
        else if (k==nz) { eb=IEY(i,j,nz);    e1=IEY(i,j,nz-1); }
        if (eb>=0) Ey[eb] = Py[e1] + cMur*(Ey[e1]-Py[eb]);
    }
    // Ez tangential on x and y faces
    if (g < uint(nEz)) {
        int i = int(g) % (nx+1);
        int j = (int(g)/(nx+1)) % (ny+1);
        int k = int(g)/((nx+1)*(ny+1));
        int eb=-1, e1=-1;
        if (i==0)       { eb=IEZ(0,j,k);     e1=IEZ(1,j,k); }
        else if (i==nx) { eb=IEZ(nx,j,k);    e1=IEZ(nx-1,j,k); }
        else if (j==0)  { eb=IEZ(i,0,k);     e1=IEZ(i,1,k); }
        else if (j==ny) { eb=IEZ(i,ny,k);    e1=IEZ(i,ny-1,k); }
        if (eb>=0) Ez[eb] = Pz[e1] + cMur*(Ez[e1]-Pz[eb]);
    }
}
)";

// surface currents (|Js| = |eta0 H_t|) and running DFT
static const char *FD_JS = R"(
layout(std430, binding=15) buffer BFM  { int faceMeta[]; }; // airCell, axis
layout(std430, binding=16) buffer BJI  { float jsInst[]; };
layout(std430, binding=17) buffer BJD  { float jsDft[]; };  // re1,im1,re2,im2
uniform int nf, doDft;
uniform float cph, sph, eta0;
void main(){
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(nf)) return;
    int i,j,k; cellCo(faceMeta[f*2u], i, j, k);
    int ax = faceMeta[f*2u+1u];
    int t1 = POLT[ax].x, t2 = POLT[ax].y;
    float h1 = eta0 * hAt(i,j,k,t1);
    float h2 = eta0 * hAt(i,j,k,t2);
    jsInst[f] = sqrt(h1*h1 + h2*h2);
    if (doDft == 1) {
        jsDft[f*4u]      += h1*cph;
        jsDft[f*4u + 1u] -= h1*sph;
        jsDft[f*4u + 2u] += h2*cph;
        jsDft[f*4u + 3u] -= h2*sph;
    }
}
)";

// Huygens surface DFT (E and eta0 H tangential phasors)
static const char *FD_HUY = R"(
layout(std430, binding=18) buffer BHM { int huyMeta[]; }; // cell, axis
layout(std430, binding=19) buffer BHD { float huyDft[]; };
uniform int nh;
uniform float cph, sph, eta0;
void main(){
    uint f = gl_GlobalInvocationID.x;
    if (f >= uint(nh)) return;
    int i,j,k; cellCo(huyMeta[f*2u], i, j, k);
    int ax = huyMeta[f*2u+1u];
    int t1 = POLT[ax].x, t2 = POLT[ax].y;
    float e1 = eAt(i,j,k,t1), e2 = eAt(i,j,k,t2);
    float m1 = eta0*hAt(i,j,k,t1), m2 = eta0*hAt(i,j,k,t2);
    uint o = f*8u;
    huyDft[o]    += e1*cph;  huyDft[o+1u] -= e1*sph;
    huyDft[o+2u] += e2*cph;  huyDft[o+3u] -= e2*sph;
    huyDft[o+4u] += m1*cph;  huyDft[o+5u] -= m1*sph;
    huyDft[o+6u] += m2*cph;  huyDft[o+7u] -= m2*sph;
}
)";

// port V/I records
static const char *FD_PORT = R"(
layout(std430, binding=20) buffer BPC { int portCells[]; };
layout(std430, binding=21) buffer BPI { ivec4 portInfo[]; }; // off,count,pol,mid
layout(std430, binding=22) buffer BPR { float portRec[]; };
uniform int numPorts, stepIdx;
uniform float dl;
void main(){
    uint p = gl_GlobalInvocationID.x;
    if (p >= uint(numPorts)) return;
    ivec4 inf = portInfo[p];
    float vSum = 0.0;
    for (int c = 0; c < inf.y; ++c) {
        int i,j,k; cellCo(portCells[inf.x + c], i, j, k);
        vSum += -eAt(i,j,k,inf.z) * dl;
    }
    int a = inf.z, t1 = CYC[a].x, t2 = CYC[a].y;
    ivec3 d1 = ivec3(0), d2 = ivec3(0);
    d1[t1] = 1; d2[t2] = 1;
    int i,j,k; cellCo(inf.w, i, j, k);
    float iVal = 2.0*dl * (
        hAt(i+d1.x, j+d1.y, k+d1.z, t2) - hAt(i-d1.x, j-d1.y, k-d1.z, t2) -
        hAt(i+d2.x, j+d2.y, k+d2.z, t1) + hAt(i-d2.x, j-d2.y, k-d2.z, t1));
    portRec[(uint(stepIdx)*uint(numPorts) + p)*2u]      = vSum;
    portRec[(uint(stepIdx)*uint(numPorts) + p)*2u + 1u] = iVal;
}
)";

// |E| cut plane
static const char *FD_PLANE = R"(
layout(std430, binding=24) buffer BPL { float plane[]; };
uniform int pAxis, pIdx, pN1, pN2;
uniform float dl;
void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uint(pN1*pN2)) return;
    int q1 = int(g) % pN1, q2 = int(g) / pN1;
    int a1 = (pAxis==0)?1:0;
    int a2 = (pAxis==2)?1:2;
    ivec3 co; co[pAxis]=pIdx; co[a1]=q1; co[a2]=q2;
    float ex=eAt(co.x,co.y,co.z,0), ey=eAt(co.x,co.y,co.z,1), ez=eAt(co.x,co.y,co.z,2);
    plane[g] = sqrt(ex*ex+ey*ey+ez*ez) * dl;
}
)";

// total field energy: sum(E^2) + eta0^2 sum(H^2)
static const char *FD_ERG = R"(
layout(std430, binding=0) buffer BEx { float Ex[]; };
layout(std430, binding=1) buffer BEy { float Ey[]; };
layout(std430, binding=2) buffer BEz { float Ez[]; };
layout(std430, binding=3) buffer BHx { float Hx[]; };
layout(std430, binding=4) buffer BHy { float Hy[]; };
layout(std430, binding=5) buffer BHz { float Hz[]; };
layout(std430, binding=23) buffer BERG { float erg[]; };
uniform int nEx, nEy, nEz, nHx, nHy, nHz, np;
uniform float eta2;
void main(){
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(np)) return;
    float s = 0.0;
    for (uint i=tid; i<uint(nEx); i+=uint(np)) s += Ex[i]*Ex[i];
    for (uint i=tid; i<uint(nEy); i+=uint(np)) s += Ey[i]*Ey[i];
    for (uint i=tid; i<uint(nEz); i+=uint(np)) s += Ez[i]*Ez[i];
    for (uint i=tid; i<uint(nHx); i+=uint(np)) s += eta2*Hx[i]*Hx[i];
    for (uint i=tid; i<uint(nHy); i+=uint(np)) s += eta2*Hy[i]*Hy[i];
    for (uint i=tid; i<uint(nHz); i+=uint(np)) s += eta2*Hz[i]*Hz[i];
    erg[tid] = s;
}
)";

//---------------------------------------------------------------------------
bool RunGpuFdtd(FdtdSolver &s, std::string &msg)
{
    const int nx = s.g.nx, ny = s.g.ny, nz = s.g.nz;
    const long long ncell = (long long)nx * ny * nz;
    if (ncell <= 0) { msg = "empty grid"; return false; }
    if (ncell > 40000000LL)  // VRAM-bound; buffer alloc below falls back if it won't fit
    {
        msg = "grid too large for the GPU FDTD path (>40 M cells)";
        return false;
    }

    GpuCompute gc;
    if (!gc.ok()) { msg = gc.error(); return false; }

    auto prog = [&](const char *body, bool fields) -> unsigned
    {
        std::string src = std::string(FD_HDR);
        if (fields) src += FD_FIELDS;
        src += body;
        return gc.buildProgram(src.c_str());
    };
    unsigned pInj = prog(FD_INJECT, false); if (!pInj) { msg = gc.error(); return false; }
    unsigned pUH  = prog(FD_UPDH,   false); if (!pUH)  { msg = gc.error(); return false; }
    unsigned pCp  = prog(FD_COPY,   false); if (!pCp)  { msg = gc.error(); return false; }
    unsigned pUE  = prog(FD_UPDE,   false); if (!pUE)  { msg = gc.error(); return false; }
    unsigned pMur = prog(FD_MUR,    false); if (!pMur) { msg = gc.error(); return false; }
    unsigned pJs  = prog(FD_JS,     true);  if (!pJs)  { msg = gc.error(); return false; }
    unsigned pHuy = prog(FD_HUY,    true);  if (!pHuy) { msg = gc.error(); return false; }
    unsigned pPort= prog(FD_PORT,   true);  if (!pPort){ msg = gc.error(); return false; }
    unsigned pPln = prog(FD_PLANE,  true);  if (!pPln) { msg = gc.error(); return false; }
    unsigned pErg = prog(FD_ERG,    false); if (!pErg) { msg = gc.error(); return false; }

    // ---- array sizes ----
    const int nEx = nx*(ny+1)*(nz+1), nEy = (nx+1)*ny*(nz+1), nEz = (nx+1)*(ny+1)*nz;
    const int nHx = (nx+1)*ny*nz, nHy = nx*(ny+1)*nz, nHz = nx*ny*(nz+1);
    const int maxE = std::max(nEx, std::max(nEy, nEz));
    const int maxH = std::max(nHx, std::max(nHy, nHz));

    // ---- per-edge update coefficients (vec2 ca,cb) ----
    auto packC = [&](const std::vector<uint8_t> &mask,
                     const std::vector<float> &ca, const std::vector<float> &cb,
                     int n, std::vector<float> &out)
    {
        out.assign((size_t)n * 2, 0.0f);
        for (int i = 0; i < n; ++i)
        {
            if (!mask[i]) continue;               // PEC -> (0,0)
            if (s.haveDiel) { out[i*2] = ca[i]; out[i*2+1] = cb[i]; }
            else            { out[i*2] = 1.0f;  out[i*2+1] = s.cb0; }
        }
    };
    std::vector<float> cEx, cEy, cEz;
    packC(s.mEx, s.caEx, s.cbEx, nEx, cEx);
    packC(s.mEy, s.caEy, s.cbEy, nEy, cEy);
    packC(s.mEz, s.caEz, s.cbEz, nEz, cEz);

    // ---- combined source list -> per-edge source coefficients ----
    struct Src { int cell, pol; float amp; };
    std::vector<Src> srcs;
    for (const auto &sc : s.sources)
        srcs.push_back({ (int)sc.cell, sc.polAxis, sc.amp });
    for (const auto &p : s.portList)
        for (size_t c : p.cells)
            srcs.push_back({ (int)c, p.polAxis, p.amp });
    std::vector<float> Sx(nEx, 0.0f), Sy(nEy, 0.0f), Sz(nEz, 0.0f);
    {
        auto add = [&](std::vector<float> &S, const std::vector<uint8_t> &mask,
                       int idx, float q)
        {
            if (mask[idx]) S[idx] += q;
        };
        for (const auto &sc : srcs)
        {
            int i = sc.cell % nx, j = (sc.cell/nx) % ny, k = sc.cell/(nx*ny);
            float q = -0.25f * sc.amp;
            auto iEx = [&](int a,int b,int c){ return (c*(ny+1)+b)*nx + a; };
            auto iEy = [&](int a,int b,int c){ return (c*ny+b)*(nx+1) + a; };
            auto iEz = [&](int a,int b,int c){ return (c*(ny+1)+b)*(nx+1) + a; };
            if (sc.pol == 0) {
                add(Sx, s.mEx, iEx(i,j,k),     q); add(Sx, s.mEx, iEx(i,j+1,k),   q);
                add(Sx, s.mEx, iEx(i,j,k+1),   q); add(Sx, s.mEx, iEx(i,j+1,k+1), q);
            } else if (sc.pol == 1) {
                add(Sy, s.mEy, iEy(i,j,k),     q); add(Sy, s.mEy, iEy(i+1,j,k),   q);
                add(Sy, s.mEy, iEy(i,j,k+1),   q); add(Sy, s.mEy, iEy(i+1,j,k+1), q);
            } else {
                add(Sz, s.mEz, iEz(i,j,k),     q); add(Sz, s.mEz, iEz(i+1,j,k),   q);
                add(Sz, s.mEz, iEz(i,j+1,k),   q); add(Sz, s.mEz, iEz(i+1,j+1,k), q);
            }
        }
    }

    // ---- buffers ----
    gc.glErr();
    gc.makeBuffer(0, (size_t)nEx*4, nullptr);
    gc.makeBuffer(1, (size_t)nEy*4, nullptr);
    gc.makeBuffer(2, (size_t)nEz*4, nullptr);
    gc.makeBuffer(3, (size_t)nHx*4, nullptr);
    gc.makeBuffer(4, (size_t)nHy*4, nullptr);
    gc.makeBuffer(5, (size_t)nHz*4, nullptr);
    gc.makeBuffer(6, cEx.size()*4, cEx.data());
    gc.makeBuffer(7, cEy.size()*4, cEy.data());
    gc.makeBuffer(8, cEz.size()*4, cEz.data());
    gc.makeBuffer(9,  (size_t)nEx*4, Sx.data());
    gc.makeBuffer(10, (size_t)nEy*4, Sy.data());
    gc.makeBuffer(11, (size_t)nEz*4, Sz.data());
    unsigned bPx = gc.makeBuffer(12, (size_t)nEx*4, nullptr);
    unsigned bPy = gc.makeBuffer(13, (size_t)nEy*4, nullptr);
    unsigned bPz = gc.makeBuffer(14, (size_t)nEz*4, nullptr);
    (void)bPx; (void)bPy; (void)bPz;

    const int nf = (int)s.surfFaces.size();
    {
        std::vector<int> meta(std::max(1, nf) * 2);
        for (int i = 0; i < nf; ++i)
        { meta[i*2] = s.surfFaces[i].airCell; meta[i*2+1] = s.surfFaces[i].axis; }
        gc.makeBuffer(15, meta.size()*4, meta.data());
    }
    unsigned bJsI = gc.makeBuffer(16, (size_t)std::max(1,nf)*4, nullptr);
    std::vector<float> zJs((size_t)std::max(1,nf)*4, 0.0f);
    unsigned bJsD = gc.makeBuffer(17, zJs.size()*4, zJs.data());

    const int nh = (int)s.huyFaces.size();
    {
        std::vector<int> meta(std::max(1, nh) * 2);
        for (int i = 0; i < nh; ++i)
        { meta[i*2] = s.huyFaces[i].cell; meta[i*2+1] = s.huyFaces[i].axis; }
        gc.makeBuffer(18, meta.size()*4, meta.data());
    }
    std::vector<float> zHuy((size_t)std::max(1,nh)*8, 0.0f);
    unsigned bHuyD = gc.makeBuffer(19, zHuy.size()*4, zHuy.data());

    const int numPorts = (int)s.portList.size();
    {
        std::vector<int> cellsFlat, info;
        for (const auto &p : s.portList)
        {
            int off = (int)cellsFlat.size();
            for (size_t c : p.cells) cellsFlat.push_back((int)c);
            int mid = p.cells.empty() ? 0 : (int)p.cells[p.cells.size()/2];
            info.push_back(off); info.push_back((int)p.cells.size());
            info.push_back(p.polAxis); info.push_back(mid);
        }
        if (cellsFlat.empty()) cellsFlat.push_back(0);
        if (info.empty()) info.assign(4, 0);
        gc.makeBuffer(20, cellsFlat.size()*4, cellsFlat.data());
        gc.makeBuffer(21, info.size()*4, info.data());
    }
    size_t portRecCount = (size_t)std::max(1, numPorts) * s.config.totalSteps * 2;
    unsigned bPortRec = gc.makeBuffer(22, portRecCount*4, nullptr);

    const int NP = 1024;
    unsigned bErg = gc.makeBuffer(23, (size_t)NP*4, nullptr);

    int planeMax = std::max(nx*ny, std::max(ny*nz, nx*nz));
    unsigned bPlane = gc.makeBuffer(24, (size_t)planeMax*4, nullptr);

    if (gc.glErr() != 0) { msg = "GPU buffer allocation failed"; return false; }

    // ---- uniforms shared by every program: dims ----
    unsigned progs[] = { pInj, pUH, pCp, pUE, pMur, pJs, pHuy, pPort, pPln, pErg };
    for (unsigned pr : progs)
    {
        gc.use(pr);
        int loc = gc.uniform(pr, "dims");
        if (loc >= 0) gc.set3i(loc, nx, ny, nz);
    }
    auto setE = [&](unsigned pr)
    {
        gc.use(pr);
        gc.set1i(gc.uniform(pr, "nEx"), nEx);
        gc.set1i(gc.uniform(pr, "nEy"), nEy);
        gc.set1i(gc.uniform(pr, "nEz"), nEz);
    };
    setE(pInj); setE(pCp); setE(pUE);
    gc.use(pMur);
    gc.set1i(gc.uniform(pMur, "nEx"), nEx);
    gc.set1i(gc.uniform(pMur, "nEy"), nEy);
    gc.set1i(gc.uniform(pMur, "nEz"), nEz);
    gc.set1f(gc.uniform(pMur, "cMur"), s.cMur);
    gc.use(pUH);
    gc.set1i(gc.uniform(pUH, "nHx"), nHx);
    gc.set1i(gc.uniform(pUH, "nHy"), nHy);
    gc.set1i(gc.uniform(pUH, "nHz"), nHz);
    gc.set1f(gc.uniform(pUH, "ch"), (float)(s.dt / (MU0 * s.g.dl)));
    gc.use(pJs);
    gc.set1i(gc.uniform(pJs, "nf"), nf);
    gc.set1f(gc.uniform(pJs, "eta0"), ETA0);
    int jsDoDft = gc.uniform(pJs, "doDft"), jsC = gc.uniform(pJs, "cph"), jsS = gc.uniform(pJs, "sph");
    gc.use(pHuy);
    gc.set1i(gc.uniform(pHuy, "nh"), nh);
    gc.set1f(gc.uniform(pHuy, "eta0"), ETA0);
    int huyC = gc.uniform(pHuy, "cph"), huyS = gc.uniform(pHuy, "sph");
    gc.use(pPort);
    gc.set1i(gc.uniform(pPort, "numPorts"), numPorts);
    gc.set1f(gc.uniform(pPort, "dl"), s.g.dl);
    int portStep = gc.uniform(pPort, "stepIdx");
    gc.use(pErg);
    gc.set1i(gc.uniform(pErg, "nEx"), nEx); gc.set1i(gc.uniform(pErg, "nEy"), nEy);
    gc.set1i(gc.uniform(pErg, "nEz"), nEz); gc.set1i(gc.uniform(pErg, "nHx"), nHx);
    gc.set1i(gc.uniform(pErg, "nHy"), nHy); gc.set1i(gc.uniform(pErg, "nHz"), nHz);
    gc.set1i(gc.uniform(pErg, "np"), NP);
    gc.set1f(gc.uniform(pErg, "eta2"), ETA0 * ETA0);

    int injSv = (gc.use(pInj), gc.uniform(pInj, "sv"));

    // cut plane geometry
    int pa = s.planeAxis.load();
    int planeN1 = 0, planeN2 = 0, planeIdx = 0;
    if (pa >= 0)
    {
        const int nn[3] = { nx, ny, nz };
        int a1 = (pa == 0) ? 1 : 0, a2 = (pa == 2) ? 1 : 2;
        planeN1 = nn[a1]; planeN2 = nn[a2];
        planeIdx = std::min(std::max(s.planeIndex, 0), nn[pa] - 1);
        gc.use(pPln);
        gc.set1i(gc.uniform(pPln, "pAxis"), pa);
        gc.set1i(gc.uniform(pPln, "pIdx"), planeIdx);
        gc.set1i(gc.uniform(pPln, "pN1"), planeN1);
        gc.set1i(gc.uniform(pPln, "pN2"), planeN2);
        gc.set1f(gc.uniform(pPln, "dl"), s.g.dl);
    }

    // ---- time loop ----
    const int total = s.config.totalSteps;
    const int settle = s.config.settleSteps;
    const int recEvery = s.config.recordEvery;
    int dftSamples = 0, huySamples = 0;
    std::vector<float> ergPart(NP), jsTmp(std::max(1, nf)), planeTmp(planeMax);

    for (int n = 0; n < total && !s.stopFlag; ++n)
    {
        float sv = s.waveformValue(n);
        bool doDft = (n >= settle);
        float phase = 2.0f * (float)M_PI * s.config.f0 * (n * s.dt);
        float cph = std::cos(phase), sph = std::sin(phase);

        // inject
        gc.use(pInj); gc.set1f(injSv, sv); gc.dispatch(maxE); gc.barrier();

        // monitors (on post-inject fields, before the field update)
        gc.use(pJs);
        gc.set1i(jsDoDft, doDft ? 1 : 0);
        gc.set1f(jsC, cph); gc.set1f(jsS, sph);
        gc.dispatch(nf); gc.barrier();
        if (doDft && nh > 0)
        {
            gc.use(pHuy); gc.set1f(huyC, cph); gc.set1f(huyS, sph);
            gc.dispatch(nh); gc.barrier();
            ++huySamples;
        }
        if (numPorts > 0)
        {
            gc.use(pPort); gc.set1i(portStep, n); gc.dispatch(numPorts); gc.barrier();
        }
        if (doDft) ++dftSamples;

        if ((n & 15) == 0)
        {
            gc.use(pErg); gc.dispatchGroups((NP + 63) / 64); gc.barrier();
            gc.readBuffer(bErg, (size_t)NP*4, ergPart.data());
            double e = 0; for (int i = 0; i < NP; ++i) e += ergPart[i];
            s.energy = (float)e;
            std::lock_guard<std::mutex> lk(s.vizMutex);
            s.energySteps.push_back(n);
            s.energyVals.push_back((float)e);
        }

        if (recEvery > 0 && (n % recEvery) == 0)
        {
            VizFrame fr;
            fr.step = n;
            gc.readBuffer(bJsI, (size_t)nf*4, jsTmp.data());
            fr.js.assign(jsTmp.begin(), jsTmp.begin() + nf);
            if (pa >= 0)
            {
                gc.use(pPln); gc.dispatch((size_t)planeN1*planeN2); gc.barrier();
                gc.readBuffer(bPlane, (size_t)planeN1*planeN2*4, planeTmp.data());
                fr.plane.assign(planeTmp.begin(), planeTmp.begin() + planeN1*planeN2);
                fr.planeAxis = pa; fr.planeIdx = planeIdx;
                fr.planeN1 = planeN1; fr.planeN2 = planeN2;
            }
            std::lock_guard<std::mutex> lk(s.vizMutex);
            s.frames.push_back(std::move(fr));
        }

        // field update: H, snapshot E, E, Mur
        gc.use(pUH);  gc.dispatch(maxH); gc.barrier();
        gc.use(pCp);  gc.dispatch(maxE); gc.barrier();
        gc.use(pUE);  gc.dispatch(maxE); gc.barrier();
        gc.use(pMur); gc.dispatch(maxE); gc.barrier();

        s.curStep = n + 1;
    }

    // ---- read monitor results back into the solver ----
    {
        std::lock_guard<std::mutex> lk(s.vizMutex);
        if (nf > 0)
        {
            std::vector<float> dft4((size_t)nf*4);
            gc.readBuffer(bJsD, dft4.size()*4, dft4.data());
            for (int i = 0; i < nf; ++i)
            {
                s.jsDftRe[i*2]     = dft4[i*4];
                s.jsDftIm[i*2]     = dft4[i*4+1];
                s.jsDftRe[i*2+1]   = dft4[i*4+2];
                s.jsDftIm[i*2+1]   = dft4[i*4+3];
            }
            s.jsInstant.assign(nf, 0.0f);
            gc.readBuffer(bJsI, (size_t)nf*4, s.jsInstant.data());
        }
        if (nh > 0)
            gc.readBuffer(bHuyD, s.huyDft.size()*4, s.huyDft.data());
        s.dftSamples = dftSamples;
        s.huySamples = huySamples;
        if (pa >= 0)
        {
            gc.use(pPln); gc.dispatch((size_t)planeN1*planeN2); gc.barrier();
            s.planeBuf.assign((size_t)planeN1*planeN2, 0.0f);
            gc.readBuffer(bPlane, (size_t)planeN1*planeN2*4, s.planeBuf.data());
            s.planeN1 = planeN1; s.planeN2 = planeN2;
        }
    }
    if (numPorts > 0)
    {
        int done = s.curStep;
        std::vector<float> rec((size_t)numPorts * done * 2);
        gc.readBuffer(bPortRec, rec.size()*4, rec.data());
        for (int p = 0; p < numPorts; ++p)
        {
            auto &port = s.portList[p];
            port.vRec.resize(done);
            port.iRec.resize(done);
            for (int t = 0; t < done; ++t)
            {
                port.vRec[t] = rec[((size_t)t*numPorts + p)*2];
                port.iRec[t] = rec[((size_t)t*numPorts + p)*2 + 1];
            }
        }
    }

    msg = gc.rendererName();
    return true;
}
