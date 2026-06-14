//---------------------------------------------------------------------------
// MomSurface.cpp - surface RWG EFIE Method of Moments
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "MomSurface.h"
#include "GpuMom.h"
#include <cmath>
#include <algorithm>
#include <map>

#pragma package(smart_init)

static const double C0   = 299792458.0;
static const double MU0  = 1.25663706212e-6;
static const double EPS0 = 8.8541878128e-12;

// 7-point triangle quadrature (degree 5), barycentric a,b,c with weights
static const double TQ_A[7] = {
    0.333333333333333, 0.059715871789770, 0.470142064105115,
    0.470142064105115, 0.797426985353087, 0.101286507323456,
    0.101286507323456 };
static const double TQ_B[7] = {
    0.333333333333333, 0.470142064105115, 0.059715871789770,
    0.470142064105115, 0.101286507323456, 0.797426985353087,
    0.101286507323456 };
static const double TQ_W[7] = {
    0.225000000000000, 0.132394152788506, 0.132394152788506,
    0.132394152788506, 0.125939180544827, 0.125939180544827,
    0.125939180544827 };

// 4-point Gauss-Legendre on [0,1] for the Duffy self-term
static const double DG[4] = { 0.069431844202974, 0.330009478207572,
                              0.669990521792428, 0.930568155797026 };
static const double DW[4] = { 0.173927422568727, 0.326072577431273,
                              0.326072577431273, 0.173927422568727 };

//---------------------------------------------------------------------------
MomSurface::MomSurface()
{
    int n = (int)std::thread::hardware_concurrency();
    n = std::max(1, std::min(16, n - 1));
    pool = new ThreadPool(n);
}

//---------------------------------------------------------------------------
MomSurface::~MomSurface()
{
    delete pool;
}

//---------------------------------------------------------------------------
void MomSurface::setup(const TriMesh &mesh, int prop, int pol, float freq,
                       bool gpu)
{
    propAxis = prop;
    polAxis  = pol;
    f0       = freq;
    useGpu   = gpu;
    finished = false;
    running  = false;
    stopFlag = false;
    curStep  = 0;
    usedGpu  = false;
    gpuMsg.clear();
    phaseText = "idle";

    // A dense MoM matrix is N x N (N ~ 1.5 x triangles) and the direct solve
    // is O(N^3), so a high-poly STL must be coarsened. Small meshes pass
    // through unchanged; large ones are vertex-clustered down to a tractable
    // triangle count (2000 tris -> ~3000 unknowns -> a few-second fill and a
    // ~30 s solve, plenty for an electrically small object).
    const int MAXTRI = 2000;
    decimatedFrom = 0;
    if (mesh.triCount() <= MAXTRI)
    {
        buildRwg(mesh);
    }
    else
    {
        Aabb bb;
        for (const auto &v : mesh.verts)
            bb.grow(v);
        float diag = bb.valid() ? bb.size().length() : 1.0f;
        float cell = diag * 0.01f;
        TriMesh dec;
        ClusterDecimate(mesh, cell, dec);
        int guard = 0;
        while (dec.triCount() > MAXTRI && guard++ < 30)
        {
            cell *= 1.3f;
            ClusterDecimate(mesh, cell, dec);
        }
        decimatedFrom = mesh.triCount();
        buildRwg(dec);
    }
}

//---------------------------------------------------------------------------
static inline Vec3 triVert(const std::vector<float> &v, int idx)
{
    return Vec3(v[idx*3], v[idx*3+1], v[idx*3+2]);
}

void MomSurface::buildRwg(const TriMesh &mesh)
{
    verts.clear(); tris.clear(); triC.clear(); triA.clear(); rwg.clear();

    // weld coincident vertices so adjacent triangles share indices
    Aabb bb;
    for (const auto &v : mesh.verts)
        bb.grow(v);
    double diag = bb.valid() ? bb.size().length() : 1.0;
    double tol = std::max(1e-9, diag * 1e-5);
    double inv = 1.0 / tol;
    std::map<long long, int> vhash;
    std::vector<int> remap(mesh.verts.size());
    auto vkey = [&](const Vec3 &p) -> long long
    {
        long long x = (long long)std::llround(p.x * inv);
        long long y = (long long)std::llround(p.y * inv);
        long long z = (long long)std::llround(p.z * inv);
        return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
    };
    for (size_t i = 0; i < mesh.verts.size(); ++i)
    {
        long long kk = vkey(mesh.verts[i]);
        auto it = vhash.find(kk);
        if (it != vhash.end())
            remap[i] = it->second;
        else
        {
            int id = (int)(verts.size() / 3);
            vhash[kk] = id;
            remap[i] = id;
            verts.push_back(mesh.verts[i].x);
            verts.push_back(mesh.verts[i].y);
            verts.push_back(mesh.verts[i].z);
        }
    }
    int nt = mesh.triCount();
    tris.resize(nt * 3);
    triC.resize(nt);
    triA.resize(nt);
    for (int t = 0; t < nt; ++t)
    {
        int a = remap[mesh.idx[t*3]], b = remap[mesh.idx[t*3+1]],
            c = remap[mesh.idx[t*3+2]];
        tris[t*3] = a; tris[t*3+1] = b; tris[t*3+2] = c;
        Vec3 va = triVert(verts, a), vb = triVert(verts, b), vc = triVert(verts, c);
        triC[t] = (va + vb + vc) * (1.0f / 3.0f);
        triA[t] = 0.5f * (vb - va).cross(vc - va).length();
    }

    // edge -> list of (tri, opposite-vertex). Interior edge (2 tris) = RWG.
    std::map<std::pair<int,int>, std::vector<std::pair<int,int>>> em;
    for (int t = 0; t < nt; ++t)
        for (int e = 0; e < 3; ++e)
        {
            int v0 = tris[t*3 + e], v1 = tris[t*3 + (e+1)%3];
            int opp = tris[t*3 + (e+2)%3];
            std::pair<int,int> key(std::min(v0,v1), std::max(v0,v1));
            em[key].push_back({ t, opp });
        }
    for (auto &kv : em)
    {
        if (kv.second.size() != 2)
            continue;            // boundary or non-manifold edge -> skip
        Rwg r;
        r.triP = kv.second[0].first;
        r.triM = kv.second[1].first;
        r.vP   = triVert(verts, kv.second[0].second);
        r.vM   = triVert(verts, kv.second[1].second);
        Vec3 ea = triVert(verts, kv.first.first);
        Vec3 eb = triVert(verts, kv.first.second);
        r.len  = (eb - ea).length();
        rwg.push_back(r);
    }
    N = (int)rwg.size();
}

//---------------------------------------------------------------------------
// ∫_t G dA' and ∫_t r' G dA' at field point fp.  G = exp(-jkR)/R.
//---------------------------------------------------------------------------
void MomSurface::triIntegral(const Vec3 &fp, int t, bool self,
                             cplx &Ig, cplx Igv[3]) const
{
    const double k = 2.0 * M_PI * f0 / C0;
    Ig = cplx(0,0); Igv[0] = Igv[1] = Igv[2] = cplx(0,0);
    Vec3 v0 = triVert(verts, tris[t*3]);
    Vec3 v1 = triVert(verts, tris[t*3+1]);
    Vec3 v2 = triVert(verts, tris[t*3+2]);

    auto accum = [&](const Vec3 &rp, double wArea)
    {
        double dx = fp.x-rp.x, dy = fp.y-rp.y, dz = fp.z-rp.z;
        double R = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (R < 1e-12) return;
        cplx G = std::exp(cplx(0, -k*R)) / R;
        Ig += wArea * G;
        Igv[0] += wArea * (double)rp.x * G;
        Igv[1] += wArea * (double)rp.y * G;
        Igv[2] += wArea * (double)rp.z * G;
    };

    if (!self)
    {
        double A = triA[t];
        for (int q = 0; q < 7; ++q)
        {
            double a = TQ_A[q], b = TQ_B[q], c = 1.0 - a - b;
            Vec3 rp = v0*(float)a + v1*(float)b + v2*(float)c;
            accum(rp, TQ_W[q] * A);
        }
    }
    else
    {
        // Duffy: split into 3 sub-triangles from the centroid (= fp here),
        // singular vertex at the centroid; Jacobian ~ u removes 1/R.
        Vec3 C = triC[t];
        Vec3 corner[3] = { v0, v1, v2 };
        for (int s = 0; s < 3; ++s)
        {
            Vec3 B = corner[s], D = corner[(s+1)%3];
            double subA = 0.5 * (B - C).cross(D - C).length();
            for (int iu = 0; iu < 4; ++iu)
            {
                double u = DG[iu];
                for (int iv = 0; iv < 4; ++iv)
                {
                    double vv = DG[iv];
                    Vec3 rp = C + ((B - C)*(float)(1.0-vv) + (D - C)*(float)vv)
                                  * (float)u;
                    // dA' = 2 subA * u du dv
                    accum(rp, DW[iu]*DW[iv] * 2.0 * subA * u);
                }
            }
        }
    }
}

//---------------------------------------------------------------------------
// RWG EFIE entry (centroid testing)
//---------------------------------------------------------------------------
MomSurface::cplx MomSurface::zEntry(int m, int n) const
{
    const double w  = 2.0 * M_PI * f0;
    const double k  = w / C0;
    const double fourPi = 4.0 * M_PI;
    const Rwg &M = rwg[m], &Nn = rwg[n];

    int mTri[2] = { M.triP, M.triM };
    Vec3 mFree[2] = { M.vP, M.vM };
    double mSgn[2] = { +1.0, -1.0 };
    int nTri[2] = { Nn.triP, Nn.triM };
    Vec3 nFree[2] = { Nn.vP, Nn.vM };
    double nSgn[2] = { +1.0, -1.0 };

    cplx Z(0,0);
    // Phi^- - Phi^+ accumulation and A·rho accumulation
    for (int sp = 0; sp < 2; ++sp)
    {
        Vec3 rc = triC[mTri[sp]];
        Vec3 rhoM = (mFree[sp] - rc) * (float)(-mSgn[sp]);  // sgn*(rc-vfree)
        cplx Avec[3] = { cplx(0,0), cplx(0,0), cplx(0,0) };
        cplx Phi(0,0);
        for (int sq = 0; sq < 2; ++sq)
        {
            bool self = (mTri[sp] == nTri[sq]);
            cplx Ig; cplx Igv[3];
            triIntegral(rc, nTri[sq], self, Ig, Igv);
            double An = triA[nTri[sq]];
            double cf = Nn.len / (2.0 * An) * nSgn[sq];
            // ∫ f_n G = cf * (Igv - vfree*Ig)
            Avec[0] += (MU0/fourPi) * cf * (Igv[0] - (double)nFree[sq].x*Ig);
            Avec[1] += (MU0/fourPi) * cf * (Igv[1] - (double)nFree[sq].y*Ig);
            Avec[2] += (MU0/fourPi) * cf * (Igv[2] - (double)nFree[sq].z*Ig);
            // charge term: div f_n = sgn * len/A
            double dn = nSgn[sq] * Nn.len / An;
            Phi += -(1.0/(cplx(0,w*EPS0))) * (1.0/fourPi) * dn * Ig;
        }
        cplx AdotRho = Avec[0]*(double)rhoM.x + Avec[1]*(double)rhoM.y
                     + Avec[2]*(double)rhoM.z;
        // Z += l_m [ jw A·rho/2  +/- Phi ]   (+Phi for sp=- , -Phi for sp=+)
        cplx jw(0, w);
        Z += (double)M.len * ( jw * AdotRho * 0.5
                               + (sp == 1 ? Phi : -Phi) );
    }
    return Z;
}

//---------------------------------------------------------------------------
void MomSurface::fillCpu()
{
    Z.assign((size_t)N * N, cplx(0,0));
    int chunks = pool->threadCount();
    pool->run(chunks, [&](int c)
    {
        int m0 = (int)((long long)N * c / chunks);
        int m1 = (int)((long long)N * (c + 1) / chunks);
        for (int m = m0; m < m1; ++m)
            for (int n = m; n < N; ++n)
            {
                cplx z = zEntry(m, n);
                Z[(size_t)m*N + n] = z;
                Z[(size_t)n*N + m] = z;   // Galerkin -> symmetric
            }
    });
}

//---------------------------------------------------------------------------
void MomSurface::excite()
{
    const double k = 2.0 * M_PI * f0 / C0;
    Vec3 kh(0,0,0); kh.set(propAxis, 1.0f);    // propagation +axis
    Vec3 eh(0,0,0); eh.set(polAxis, 1.0f);     // E polarization
    V.assign(N, cplx(0,0));
    for (int m = 0; m < N; ++m)
    {
        const Rwg &M = rwg[m];
        int mTri[2] = { M.triP, M.triM };
        Vec3 mFree[2] = { M.vP, M.vM };
        double mSgn[2] = { +1.0, -1.0 };
        cplx v(0,0);
        for (int sp = 0; sp < 2; ++sp)
        {
            Vec3 rc = triC[mTri[sp]];
            Vec3 rho = (mFree[sp] - rc) * (float)(-mSgn[sp]);
            double phase = -k * (kh.x*rc.x + kh.y*rc.y + kh.z*rc.z);
            cplx Einc = std::exp(cplx(0, phase));   // unit amplitude
            double edotrho = eh.x*rho.x + eh.y*rho.y + eh.z*rho.z;
            v += 0.5 * (double)M.len * edotrho * Einc;
        }
        V[m] = v;
    }
}

//---------------------------------------------------------------------------
void MomSurface::solveCpu()
{
    std::vector<cplx> A = Z;
    J = V;
    for (int col = 0; col < N; ++col)
    {
        int piv = col; double best = std::abs(A[(size_t)col*N+col]);
        for (int r = col+1; r < N; ++r)
        {
            double v = std::abs(A[(size_t)r*N+col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col)
        {
            for (int j = 0; j < N; ++j)
                std::swap(A[(size_t)col*N+j], A[(size_t)piv*N+j]);
            std::swap(J[col], J[piv]);
        }
        cplx d = A[(size_t)col*N+col];
        if (std::abs(d) < 1e-300) continue;
        for (int r = col+1; r < N; ++r)
        {
            cplx f = A[(size_t)r*N+col] / d;
            for (int j = col; j < N; ++j)
                A[(size_t)r*N+j] -= f * A[(size_t)col*N+j];
            J[r] -= f * J[col];
        }
    }
    for (int r = N-1; r >= 0; --r)
    {
        cplx s = J[r];
        for (int j = r+1; j < N; ++j)
            s -= A[(size_t)r*N+j] * J[j];
        cplx d = A[(size_t)r*N+r];
        J[r] = (std::abs(d) > 1e-300) ? s / d : cplx(0,0);
    }
}

//---------------------------------------------------------------------------
void MomSurface::postProcess()
{
    int nt = (int)triA.size();
    triMag.assign(nt, 0.0f);
    // J(centroid of tri) = sum over its RWG edges of I_m f_m(centroid)
    std::vector<cplx> jx(nt, 0), jy(nt, 0), jz(nt, 0);
    for (int m = 0; m < N; ++m)
    {
        const Rwg &M = rwg[m];
        int t[2] = { M.triP, M.triM };
        Vec3 vf[2] = { M.vP, M.vM };
        double sg[2] = { +1.0, -1.0 };
        for (int s = 0; s < 2; ++s)
        {
            Vec3 rc = triC[t[s]];
            double cf = M.len / (2.0 * triA[t[s]]) * sg[s];
            Vec3 rho = (rc - vf[s]);
            jx[t[s]] += J[m] * (cf * rho.x);
            jy[t[s]] += J[m] * (cf * rho.y);
            jz[t[s]] += J[m] * (cf * rho.z);
        }
    }
    for (int t = 0; t < nt; ++t)
        triMag[t] = (float)std::sqrt(std::norm(jx[t]) + std::norm(jy[t])
                                     + std::norm(jz[t]));
}

//---------------------------------------------------------------------------
void MomSurface::run()
{
    running = true;
    if (N == 0)
    {
        phaseText = "no RWG basis (need a closed/manifold triangle mesh)";
        finished = true; running = false;
        return;
    }
    // dense matrix guard: never attempt an allocation that would fail.
    // N*N complex doubles (CPU) = 16*N^2 bytes. Cap at a safe size.
    if ((double)N * N * 16.0 > 4.0e9)   // ~4 GB matrix
    {
        phaseText = "mesh too detailed for MoM";
        finished = true; running = false;
        return;
    }
    maxStep = N;
    phaseText = "excitation";
    excite();
    usedGpu = false;
    if (useGpu)
    {
        phaseText = "fill + solve (GPU)";
        usedGpu = RunGpuMomSurf(*this, gpuMsg);
    }
    if (!usedGpu)
    {
        phaseText = "filling matrix (CPU)";
        fillCpu();
        phaseText = "solving (CPU)";
        solveCpu();
    }
    curStep = maxStep;
    phaseText = "post-processing";
    postProcess();
    phaseText = "done";
    finished = true;
    running  = false;
}

//---------------------------------------------------------------------------
// monostatic radar cross section (m^2): backscatter toward the source.
// sigma = (k^2 eta^2 / 4pi) |N_perp|^2 with unit incident amplitude.
//---------------------------------------------------------------------------
double MomSurface::monostaticRcsM2() const
{
    if (N == 0 || J.empty())
        return 0.0;
    const double k = 2.0 * M_PI * f0 / C0;
    const double eta = 376.730313;
    int nt = (int)triA.size();
    std::vector<cplx> mx(nt,0), my(nt,0), mz(nt,0);
    for (int m = 0; m < N; ++m)
    {
        const Rwg &M = rwg[m];
        int t[2] = { M.triP, M.triM };
        Vec3 vf[2] = { M.vP, M.vM };
        double sg[2] = { +1.0, -1.0 };
        for (int s = 0; s < 2; ++s)
        {
            Vec3 rc = triC[t[s]];
            double cf = M.len * sg[s] * 0.5;   // (len/(2A))*sg*A
            Vec3 rho = rc - vf[s];
            mx[t[s]] += J[m]*(cf*rho.x);
            my[t[s]] += J[m]*(cf*rho.y);
            mz[t[s]] += J[m]*(cf*rho.z);
        }
    }
    Vec3 rh(0,0,0);
    rh.set(propAxis, -1.0f);            // backscatter = -incidence
    cplx Nx(0,0), Ny(0,0), Nz(0,0);
    for (int t = 0; t < nt; ++t)
    {
        double phase = k*(rh.x*triC[t].x + rh.y*triC[t].y + rh.z*triC[t].z);
        cplx e = std::exp(cplx(0, phase));
        Nx += mx[t]*e; Ny += my[t]*e; Nz += mz[t]*e;
    }
    cplx Ndr = Nx*(double)rh.x + Ny*(double)rh.y + Nz*(double)rh.z;
    double nperp2 = std::norm(Nx)+std::norm(Ny)+std::norm(Nz) - std::norm(Ndr);
    if (nperp2 < 0) nperp2 = 0;
    return (k*k*eta*eta/(4.0*M_PI)) * nperp2;
}

//---------------------------------------------------------------------------
void MomSurface::runRcsSweep(const std::vector<float> &freqs)
{
    running  = true;
    finished = false;
    stopFlag = false;
    sweepFreqs = freqs;
    sweepRcs.assign(freqs.size(), 0.0);
    maxStep = (int)freqs.size();
    curStep = 0;
    for (size_t i = 0; i < freqs.size() && !stopFlag; ++i)
    {
        f0 = freqs[i];
        phaseText = "sweep: excitation";
        excite();
        usedGpu = false;
        if (useGpu)
        {
            phaseText = "sweep: fill + solve (GPU)";
            usedGpu = RunGpuMomSurf(*this, gpuMsg);
        }
        if (!usedGpu)
        {
            phaseText = "sweep: fill + solve (CPU)";
            fillCpu();
            solveCpu();
        }
        sweepRcs[i] = monostaticRcsM2();
        curStep = (int)i + 1;
    }
    postProcess();        // leave the last frequency's currents displayable
    phaseText = "sweep done";
    finished = true;
    running  = false;
}

void MomSurface::getSweep(std::vector<float> &freqs, std::vector<double> &rcs)
{
    freqs = sweepFreqs;
    rcs   = sweepRcs;
}

//---------------------------------------------------------------------------
void MomSurface::getTriCurrents(std::vector<Vec3> &v, std::vector<int> &idx,
                                std::vector<float> &mag)
{
    v.resize(verts.size() / 3);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = triVert(verts, (int)i);
    idx = tris;
    mag = triMag;
}

//---------------------------------------------------------------------------
float MomSurface::matrixSymmetryError()
{
    if (Z.empty()) return 1.0f;
    double num = 0, den = 0;
    for (int m = 0; m < N; ++m)
        for (int n = 0; n < N; ++n)
        {
            cplx a = Z[(size_t)m*N+n], b = Z[(size_t)n*N+m];
            num += std::norm(a - b);
            den += std::norm(a);
        }
    return (den > 0) ? (float)std::sqrt(num/den) : 0.0f;
}

//---------------------------------------------------------------------------
bool MomSurface::computeFarField(FarFieldData &out, int nTheta, int nPhi)
{
    if (N == 0) return false;
    const double k = 2.0 * M_PI * f0 / C0;
    int nt = (int)triA.size();
    // per-triangle current moment (J_centroid * area)
    std::vector<cplx> mx(nt,0), my(nt,0), mz(nt,0);
    for (int m = 0; m < N; ++m)
    {
        const Rwg &M = rwg[m];
        int t[2] = { M.triP, M.triM };
        Vec3 vf[2] = { M.vP, M.vM };
        double sg[2] = { +1.0, -1.0 };
        for (int s = 0; s < 2; ++s)
        {
            Vec3 rc = triC[t[s]];
            double cf = M.len / (2.0 * triA[t[s]]) * sg[s] * triA[t[s]];
            Vec3 rho = (rc - vf[s]);
            mx[t[s]] += J[m]*(cf*rho.x); my[t[s]] += J[m]*(cf*rho.y);
            mz[t[s]] += J[m]*(cf*rho.z);
        }
    }
    out.nTheta = nTheta; out.nPhi = nPhi;
    out.U.assign((size_t)nTheta*nPhi, 0.0f);
    for (int ti = 0; ti < nTheta; ++ti)
    {
        double th = M_PI*ti/(nTheta-1), st = std::sin(th), ct = std::cos(th);
        for (int pi = 0; pi < nPhi; ++pi)
        {
            double ph = 2.0*M_PI*pi/(nPhi-1), cp = std::cos(ph), sp = std::sin(ph);
            Vec3 rh((float)(st*cp),(float)(st*sp),(float)ct);
            Vec3 thh((float)(ct*cp),(float)(ct*sp),(float)(-st));
            Vec3 phh((float)(-sp),(float)cp,0.0f);
            cplx Nth(0,0), Nph(0,0);
            for (int t = 0; t < nt; ++t)
            {
                double phase = k*(rh.x*triC[t].x + rh.y*triC[t].y + rh.z*triC[t].z);
                cplx e = std::exp(cplx(0, phase));
                cplx cx = mx[t]*e, cy = my[t]*e, cz = mz[t]*e;
                Nth += cx*(double)thh.x + cy*(double)thh.y + cz*(double)thh.z;
                Nph += cx*(double)phh.x + cy*(double)phh.y + cz*(double)phh.z;
            }
            out.U[(size_t)ti*nPhi+pi] = (float)(std::norm(Nth)+std::norm(Nph));
        }
    }
    float uMax = 0; int iMax = 0; double prad = 0;
    double dTh = M_PI/(nTheta-1), dPh = 2.0*M_PI/(nPhi-1);
    for (int ti = 0; ti < nTheta; ++ti)
    {
        double th = M_PI*ti/(nTheta-1), wTh = (ti==0||ti==nTheta-1)?0.5:1.0;
        for (int pi = 0; pi < nPhi; ++pi)
        {
            double wPh = (pi==0||pi==nPhi-1)?0.5:1.0;
            float u = out.U[(size_t)ti*nPhi+pi];
            if (u > uMax){ uMax=u; iMax=ti*nPhi+pi; }
            prad += (double)u*std::sin(th)*wTh*wPh*dTh*dPh;
        }
    }
    out.uMax = uMax;
    out.directivity = (prad>0)?(float)(4.0*M_PI*uMax/prad):0.0f;
    out.peakThetaDeg = 180.0f*(iMax/nPhi)/(nTheta-1);
    out.peakPhiDeg   = 360.0f*(iMax%nPhi)/(nPhi-1);
    return true;
}
