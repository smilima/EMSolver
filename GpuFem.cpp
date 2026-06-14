//---------------------------------------------------------------------------
// GpuFem.cpp - COCG solve of the FEM system on the GPU
//
// The assembled CSR matrix and vectors live in single-precision complex
// (vec2) SSBOs. The diagonally preconditioned COCG iteration runs entirely
// on the GPU: SpMV, complex AXPYs and the diagonal preconditioner are
// compute kernels; only the three scalar reductions per iteration (two
// complex dot products and one residual norm) are read back to drive the
// scalar recurrence on the CPU. Falls back to the CPU path on any failure.
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GpuFem.h"
#include "FemSolver.h"
#include "GpuCompute.h"
#include <vector>
#include <complex>
#include <cmath>

#pragma package(smart_init)

//---------------------------------------------------------------------------
static const char *FEM_COMMON = R"(
#version 430
layout(local_size_x = 64) in;
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
vec2 cdiv(vec2 a, vec2 b) {
    float d = b.x*b.x + b.y*b.y;
    return vec2((a.x*b.x + a.y*b.y) / d, (a.y*b.x - a.x*b.y) / d);
}
)";

// y = A x  (CSR, complex)
static const char *FEM_SPMV = R"(
layout(std430, binding = 0) buffer Bval  { vec2 val[]; };
layout(std430, binding = 1) buffer Bcol  { int  col[]; };
layout(std430, binding = 2) buffer Brow  { int  rowp[]; };
layout(std430, binding = 7) buffer Bp    { vec2 pv[]; };
layout(std430, binding = 8) buffer Bq    { vec2 qv[]; };
uniform int n;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= uint(n)) return;
    vec2 s = vec2(0.0);
    int e0 = rowp[row], e1 = rowp[row + 1u];
    for (int e = e0; e < e1; ++e)
        s += cmul(val[e], pv[col[e]]);
    qv[row] = s;
}
)";

// z = r / diag   (diagonal preconditioner)
static const char *FEM_PRECOND = R"(
layout(std430, binding = 3) buffer Bdiag { vec2 dg[]; };
layout(std430, binding = 5) buffer Br    { vec2 rv[]; };
layout(std430, binding = 6) buffer Bz    { vec2 zv[]; };
uniform int n;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(n)) return;
    zv[i] = cdiv(rv[i], dg[i]);
}
)";

// x += alpha*p ;  r -= alpha*q
static const char *FEM_XR = R"(
layout(std430, binding = 4) buffer Bx { vec2 xv[]; };
layout(std430, binding = 5) buffer Br { vec2 rv[]; };
layout(std430, binding = 7) buffer Bp { vec2 pv[]; };
layout(std430, binding = 8) buffer Bq { vec2 qv[]; };
uniform int n;
uniform float aRe;
uniform float aIm;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(n)) return;
    vec2 a = vec2(aRe, aIm);
    xv[i] += cmul(a, pv[i]);
    rv[i] -= cmul(a, qv[i]);
}
)";

// p = z + beta*p
static const char *FEM_PUPD = R"(
layout(std430, binding = 6) buffer Bz { vec2 zv[]; };
layout(std430, binding = 7) buffer Bp { vec2 pv[]; };
uniform int n;
uniform float bRe;
uniform float bIm;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(n)) return;
    vec2 b = vec2(bRe, bIm);
    pv[i] = zv[i] + cmul(b, pv[i]);
}
)";

// unconjugated complex dot of two binding-selected vectors into partials.
// 'which' picks the operand pair: 0 = (p,q), 1 = (r,z).
static const char *FEM_DOT = R"(
layout(std430, binding = 5) buffer Br  { vec2 rv[]; };
layout(std430, binding = 6) buffer Bz  { vec2 zv[]; };
layout(std430, binding = 7) buffer Bp  { vec2 pv[]; };
layout(std430, binding = 8) buffer Bq  { vec2 qv[]; };
layout(std430, binding = 9) buffer Bpc { vec2 part[]; };
uniform int n;
uniform int np;
uniform int which;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(np)) return;
    vec2 s = vec2(0.0);
    for (uint i = tid; i < uint(n); i += uint(np)) {
        vec2 a = (which == 0) ? pv[i] : rv[i];
        vec2 b = (which == 0) ? qv[i] : zv[i];
        s += cmul(a, b);
    }
    part[tid] = s;
}
)";

// sum |r|^2 into real partials
static const char *FEM_NRM = R"(
layout(std430, binding = 5)  buffer Br  { vec2 rv[]; };
layout(std430, binding = 10) buffer Bpr { float partR[]; };
uniform int n;
uniform int np;
void main() {
    uint tid = gl_GlobalInvocationID.x;
    if (tid >= uint(np)) return;
    float s = 0.0;
    for (uint i = tid; i < uint(n); i += uint(np)) {
        vec2 v = rv[i];
        s += v.x*v.x + v.y*v.y;
    }
    partR[tid] = s;
}
)";

//---------------------------------------------------------------------------
bool RunGpuFemCocg(FemSolver &s, std::string &msg)
{
    typedef std::complex<double> cplx;
    const int n = (int)s.nUnknowns;
    if (n <= 0)
    {
        msg = "no unknowns";
        return false;
    }

    GpuCompute gc;
    if (!gc.ok())
    {
        msg = gc.error();
        return false;
    }

    auto prog = [&](const char *body) -> unsigned
    {
        std::string src = std::string(FEM_COMMON) + body;
        return gc.buildProgram(src.c_str());
    };
    unsigned prSpmv = prog(FEM_SPMV);    if (!prSpmv)   { msg = gc.error(); return false; }
    unsigned prPre  = prog(FEM_PRECOND); if (!prPre)    { msg = gc.error(); return false; }
    unsigned prXr   = prog(FEM_XR);      if (!prXr)     { msg = gc.error(); return false; }
    unsigned prPupd = prog(FEM_PUPD);    if (!prPupd)   { msg = gc.error(); return false; }
    unsigned prDot  = prog(FEM_DOT);     if (!prDot)    { msg = gc.error(); return false; }
    unsigned prNrm  = prog(FEM_NRM);     if (!prNrm)    { msg = gc.error(); return false; }

    // ---- pack CSR + vectors into single-precision ----
    const int nnz = (int)s.val.size();
    std::vector<float> valf(nnz * 2);
    for (int i = 0; i < nnz; ++i)
    {
        valf[i * 2]     = (float)s.val[i].real();
        valf[i * 2 + 1] = (float)s.val[i].imag();
    }
    std::vector<float> diagf(n * 2, 0.0f);
    diagf.assign(n * 2, 0.0f);
    for (int r = 0; r < n; ++r)
    {
        diagf[r * 2] = 1.0f;                 // default if no diagonal entry
        for (int e = s.rowPtr[r]; e < s.rowPtr[r + 1]; ++e)
            if (s.colIdx[e] == r)
            {
                diagf[r * 2]     = (float)s.val[e].real();
                diagf[r * 2 + 1] = (float)s.val[e].imag();
            }
    }
    std::vector<float> rhsf(n * 2);
    for (int i = 0; i < n; ++i)
    {
        rhsf[i * 2]     = (float)s.rhs[i].real();
        rhsf[i * 2 + 1] = (float)s.rhs[i].imag();
    }

    const int NP = 1024;                     // reduction partials
    gc.glErr();
    unsigned bVal  = gc.makeBuffer(0, (size_t)nnz * 2 * sizeof(float), valf.data());
    unsigned bCol  = gc.makeBuffer(1, (size_t)nnz * sizeof(int), s.colIdx.data());
    unsigned bRow  = gc.makeBuffer(2, (size_t)(n + 1) * sizeof(int), s.rowPtr.data());
    unsigned bDiag = gc.makeBuffer(3, (size_t)n * 2 * sizeof(float), diagf.data());
    unsigned bX    = gc.makeBuffer(4, (size_t)n * 2 * sizeof(float), nullptr); // x = 0
    unsigned bR    = gc.makeBuffer(5, (size_t)n * 2 * sizeof(float), rhsf.data()); // r = b
    unsigned bZ    = gc.makeBuffer(6, (size_t)n * 2 * sizeof(float), nullptr);
    unsigned bP    = gc.makeBuffer(7, (size_t)n * 2 * sizeof(float), nullptr); // p = 0
    unsigned bQ    = gc.makeBuffer(8, (size_t)n * 2 * sizeof(float), nullptr);
    unsigned bPc   = gc.makeBuffer(9, (size_t)NP * 2 * sizeof(float), nullptr);
    unsigned bPr   = gc.makeBuffer(10, (size_t)NP * sizeof(float), nullptr);
    (void)bVal; (void)bCol; (void)bRow; (void)bDiag; (void)bX; (void)bZ; (void)bQ;
    if (gc.glErr() != 0)
    {
        msg = "GPU buffer allocation failed";
        return false;
    }

    auto setN = [&](unsigned pr)
    {
        gc.use(pr);
        gc.set1i(gc.uniform(pr, "n"), n);
    };
    setN(prSpmv); setN(prPre); setN(prXr); setN(prPupd);
    gc.use(prDot);
    gc.set1i(gc.uniform(prDot, "n"), n);
    gc.set1i(gc.uniform(prDot, "np"), NP);
    gc.use(prNrm);
    gc.set1i(gc.uniform(prNrm, "n"), n);
    gc.set1i(gc.uniform(prNrm, "np"), NP);

    std::vector<float> partC(NP * 2), partR(NP);
    auto cdot = [&](int which) -> cplx
    {
        gc.use(prDot);
        gc.set1i(gc.uniform(prDot, "which"), which);
        gc.dispatchGroups((NP + 63) / 64);
        gc.barrier();
        gc.readBuffer(bPc, (size_t)NP * 2 * sizeof(float), partC.data());
        double re = 0, im = 0;
        for (int i = 0; i < NP; ++i) { re += partC[i * 2]; im += partC[i * 2 + 1]; }
        return cplx(re, im);
    };
    auto rnorm2 = [&]() -> double
    {
        gc.use(prNrm);
        gc.dispatchGroups((NP + 63) / 64);
        gc.barrier();
        gc.readBuffer(bPr, (size_t)NP * sizeof(float), partR.data());
        double s2 = 0;
        for (int i = 0; i < NP; ++i) s2 += partR[i];
        return s2;
    };

    // bn = ||b|| ;  r already = b
    double bn = std::sqrt(rnorm2());
    if (bn < 1e-30)
    {
        msg = "zero right-hand side";
        return false;
    }

    // z = M^-1 r ; p = z (p starts at 0, so p = z + 0*p)
    gc.use(prPre);  gc.dispatch(n);  gc.barrier();
    gc.use(prPupd);
    gc.set1f(gc.uniform(prPupd, "bRe"), 0.0f);
    gc.set1f(gc.uniform(prPupd, "bIm"), 0.0f);
    gc.dispatch(n); gc.barrier();

    cplx rho = cdot(1);          // (r, z)
    const double tol = 1e-4;     // single precision floor
    const int maxIt = 8000;
    s.maxIter = maxIt;
    s.didConverge = false;

    int it = 0;
    for (; it < maxIt && !s.stopFlag; ++it)
    {
        gc.use(prSpmv); gc.dispatch(n); gc.barrier();        // q = A p
        cplx pq = cdot(0);                                   // (p, q)
        if (std::abs(pq) < 1e-30) break;
        cplx alpha = rho / pq;

        gc.use(prXr);
        gc.set1f(gc.uniform(prXr, "aRe"), (float)alpha.real());
        gc.set1f(gc.uniform(prXr, "aIm"), (float)alpha.imag());
        gc.dispatch(n); gc.barrier();                        // x+=a p; r-=a q

        double res = std::sqrt(rnorm2()) / bn;
        s.resNorm = (float)res;
        s.curIter = it + 1;
        if (res < tol) { s.didConverge = true; ++it; break; }

        gc.use(prPre); gc.dispatch(n); gc.barrier();         // z = M^-1 r
        cplx rho1 = cdot(1);                                 // (r, z)
        if (std::abs(rho) < 1e-30) break;
        cplx beta = rho1 / rho;
        rho = rho1;

        gc.use(prPupd);
        gc.set1f(gc.uniform(prPupd, "bRe"), (float)beta.real());
        gc.set1f(gc.uniform(prPupd, "bIm"), (float)beta.imag());
        gc.dispatch(n); gc.barrier();                        // p = z + beta p
    }

    // ---- read the solution back ----
    std::vector<float> xf(n * 2);
    gc.readBuffer(bX, (size_t)n * 2 * sizeof(float), xf.data());
    s.sol.assign(n, cplx(0, 0));
    for (int i = 0; i < n; ++i)
        s.sol[i] = cplx(xf[i * 2], xf[i * 2 + 1]);

    msg = gc.rendererName();
    return true;
}
