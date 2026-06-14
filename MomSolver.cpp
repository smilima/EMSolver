//---------------------------------------------------------------------------
// MomSolver.cpp - thin-wire EFIE Method of Moments
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "MomSolver.h"
#include "GpuMom.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>

#pragma package(smart_init)

static const double C0   = 299792458.0;
static const double MU0  = 1.25663706212e-6;

// 8-point Gauss-Legendre on [0,1] (weights sum to 1)
const double MOM_GU[8] = {
    0.019855071751232, 0.101666761293187, 0.237233795041836,
    0.408282678752175, 0.591717321247825, 0.762766204958164,
    0.898333238706813, 0.980144928248768 };
const double MOM_GW[8] = {
    0.050614268145188, 0.111190517226687, 0.156853322938944,
    0.181341891689181, 0.181341891689181, 0.156853322938944,
    0.111190517226687, 0.050614268145188 };

//---------------------------------------------------------------------------
MomSolver::MomSolver()
{
    int n = (int)std::thread::hardware_concurrency();
    n = std::max(1, std::min(16, n - 1));
    pool = new ThreadPool(n);
}

//---------------------------------------------------------------------------
MomSolver::~MomSolver()
{
    delete pool;
}

//---------------------------------------------------------------------------
void MomSolver::setupWire(const std::vector<std::vector<Vec3>> &pl,
                          bool feed, const Vec3 &fa, const Vec3 &fb,
                          float rad, float freq, int spl, bool gpu)
{
    polylines   = pl;
    hasFeed     = feed;
    feedA       = fa;
    feedB       = fb;
    f0          = freq;
    segPerLambda= std::max(6, spl);
    useGpu      = gpu;
    double lam  = C0 / f0;
    radius      = (rad > 0.0f) ? rad : (float)(lam / 300.0);
    finished    = false;
    running     = false;
    stopFlag    = false;
    curStep     = 0;
    zinOk       = false;
    usedGpu     = false;
    gpuMsg.clear();
    phaseText   = "idle";
}

//---------------------------------------------------------------------------
// weld + path assembly
//---------------------------------------------------------------------------
void MomSolver::assemble()
{
    nodes.clear();
    segs.clear();
    basis.clear();
    feedBasis = -1;

    double lam = C0 / f0;
    double maxLen = lam / segPerLambda;
    double weldTol = maxLen * 0.05;

    // raw subdivided segments as endpoint pairs
    std::vector<std::pair<Vec3, Vec3>> raw;
    auto addSpan = [&](const Vec3 &p0, const Vec3 &p1)
    {
        float L = (p1 - p0).length();
        if (L < 1e-9f)
            return;
        int ns = std::max(1, (int)std::ceil(L / maxLen));
        for (int s = 0; s < ns; ++s)
        {
            Vec3 a = p0 + (p1 - p0) * ((float)s / ns);
            Vec3 b = p0 + (p1 - p0) * ((float)(s + 1) / ns);
            raw.push_back({ a, b });
        }
    };
    for (const auto &poly : polylines)
        for (size_t i = 0; i + 1 < poly.size(); ++i)
            addSpan(poly[i], poly[i + 1]);
    if (hasFeed)
        addSpan(feedA, feedB);
    if (raw.empty())
        return;

    // weld nodes via a quantized hash
    std::unordered_map<long long, int> hash;
    double inv = 1.0 / std::max(weldTol, 1e-12);
    auto key = [&](const Vec3 &p) -> long long
    {
        long long x = (long long)std::llround(p.x * inv);
        long long y = (long long)std::llround(p.y * inv);
        long long z = (long long)std::llround(p.z * inv);
        return (x * 73856093LL) ^ (y * 19349663LL) ^ (z * 83492791LL);
    };
    auto getNode = [&](const Vec3 &p) -> int
    {
        long long k = key(p);
        auto it = hash.find(k);
        if (it != hash.end())
            return it->second;
        int id = (int)nodes.size();
        hash[k] = id;
        nodes.push_back(p);
        return id;
    };
    for (auto &r : raw)
    {
        int n0 = getNode(r.first), n1 = getNode(r.second);
        if (n0 == n1)
            continue;
        Seg s;
        s.n0 = n0; s.n1 = n1;
        Vec3 d = nodes[n1] - nodes[n0];
        s.len = d.length();
        s.t   = d / s.len;
        s.mid = (nodes[n0] + nodes[n1]) * 0.5f;
        segs.push_back(s);
    }
    if (segs.empty())
        return;

    // adjacency
    std::vector<std::vector<int>> inc(nodes.size());
    for (int s = 0; s < (int)segs.size(); ++s)
    {
        inc[segs[s].n0].push_back(s);
        inc[segs[s].n1].push_back(s);
    }
    std::vector<int> deg(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
        deg[i] = (int)inc[i].size();

    // walk simple paths from degree-1 endpoints
    std::vector<char> usedSeg(segs.size(), 0);
    std::vector<int> nodeBasis(nodes.size(), -1);
    auto otherEnd = [&](int s, int n) { return segs[s].n0 == n ? segs[s].n1 : segs[s].n0; };

    auto buildPath = [&](int start)
    {
        std::vector<int> path;
        path.push_back(start);
        int cur = start, prev = -1;
        for (;;)
        {
            int nextSeg = -1;
            for (int s : inc[cur])
                if (!usedSeg[s] && s != prev) { nextSeg = s; break; }
            if (nextSeg < 0)
                break;
            usedSeg[nextSeg] = 1;
            int nn = otherEnd(nextSeg, cur);
            path.push_back(nn);
            prev = nextSeg;
            cur = nn;
            if (deg[nn] != 2)
                break;            // endpoint or junction terminates the path
        }
        // internal nodes (not the two ends) -> rooftop basis
        for (size_t k = 1; k + 1 < path.size(); ++k)
        {
            int pm = path[k - 1], pc = path[k], pp = path[k + 1];
            Basis b;
            b.node = pc;
            // sub[0]: pm -> pc  (rises to node)
            Vec3 d0 = nodes[pc] - nodes[pm];
            float l0 = d0.length();
            b.sub[0] = { nodes[pm], nodes[pc], d0 / l0, l0,  1.0f / l0, 0 };
            // sub[1]: pc -> pp  (falls from node)
            Vec3 d1 = nodes[pp] - nodes[pc];
            float l1 = d1.length();
            b.sub[1] = { nodes[pc], nodes[pp], d1 / l1, l1, -1.0f / l1, 1 };
            nodeBasis[pc] = (int)basis.size();
            basis.push_back(b);
        }
    };
    for (size_t i = 0; i < nodes.size(); ++i)
        if (deg[i] == 1)
            for (int s : inc[i])
                if (!usedSeg[s])
                    buildPath((int)i);
    // any leftover (loops) - walk from an arbitrary node on them
    for (int s = 0; s < (int)segs.size(); ++s)
        if (!usedSeg[s])
            buildPath(segs[s].n0);

    N = (int)basis.size();

    // feed basis = nearest basis node to the feed midpoint
    if (hasFeed && N > 0)
    {
        Vec3 fc = (feedA + feedB) * 0.5f;
        float best = 1e30f;
        for (int b = 0; b < N; ++b)
        {
            float d = (nodes[basis[b].node] - fc).length();
            if (d < best) { best = d; feedBasis = b; }
        }
    }
}

//---------------------------------------------------------------------------
// EFIE matrix entry  Zmn  (reduced thin-wire kernel, Galerkin)
//---------------------------------------------------------------------------
MomSolver::cplx MomSolver::zEntry(int m, int n) const
{
    const double w  = 2.0 * M_PI * f0;
    const double k  = w / C0;
    const double a2 = (double)radius * radius;
    cplx s1(0, 0), s2(0, 0);
    for (int pi = 0; pi < 2; ++pi)
    {
        const SubSeg &P = basis[m].sub[pi];
        double sgnp = P.dsign * P.len;            // +-1
        for (int qi = 0; qi < 2; ++qi)
        {
            const SubSeg &Q = basis[n].sub[qi];
            double sgnq = Q.dsign * Q.len;
            double tdot = P.t.x * Q.t.x + P.t.y * Q.t.y + P.t.z * Q.t.z;
            cplx i1(0, 0), i0(0, 0);
            for (int g = 0; g < 8; ++g)
            {
                double ug = MOM_GU[g];
                Vec3 rp = P.a + (P.b - P.a) * (float)ug;
                double rampm = (P.rampType == 0) ? ug : (1.0 - ug);
                for (int h = 0; h < 8; ++h)
                {
                    double uh = MOM_GU[h];
                    Vec3 rq = Q.a + (Q.b - Q.a) * (float)uh;
                    double rampn = (Q.rampType == 0) ? uh : (1.0 - uh);
                    double dx = rp.x - rq.x, dy = rp.y - rq.y, dz = rp.z - rq.z;
                    double R = std::sqrt(dx*dx + dy*dy + dz*dz + a2);
                    double ww = MOM_GW[g] * MOM_GW[h];
                    cplx G = std::exp(cplx(0, -k * R)) / (4.0 * M_PI * R);
                    i1 += ww * rampm * rampn * G;
                    i0 += ww * G;
                }
            }
            s1 += tdot * (double)P.len * (double)Q.len * i1;
            s2 += sgnp * sgnq * i0;
        }
    }
    cplx jwm(0, w * MU0);
    return jwm * s1 - (jwm / (k * k)) * s2;
}

//---------------------------------------------------------------------------
void MomSolver::fillMatrixCpu()
{
    Z.assign((size_t)N * N, cplx(0, 0));
    int chunks = pool->threadCount();
    pool->run(chunks, [&](int c)
    {
        int m0 = (int)((long long)N * c / chunks);
        int m1 = (int)((long long)N * (c + 1) / chunks);
        for (int m = m0; m < m1; ++m)
            for (int n = m; n < N; ++n)        // symmetric: fill upper, mirror
            {
                cplx z = zEntry(m, n);
                Z[(size_t)m * N + n] = z;
                Z[(size_t)n * N + m] = z;
            }
    });
}

//---------------------------------------------------------------------------
// dense complex solve by Gaussian elimination with partial pivoting
//---------------------------------------------------------------------------
void MomSolver::solveCpu()
{
    std::vector<cplx> A = Z;          // working copy
    I = V;
    for (int col = 0; col < N; ++col)
    {
        int piv = col;
        double best = std::abs(A[(size_t)col * N + col]);
        for (int r = col + 1; r < N; ++r)
        {
            double v = std::abs(A[(size_t)r * N + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col)
        {
            for (int j = 0; j < N; ++j)
                std::swap(A[(size_t)col * N + j], A[(size_t)piv * N + j]);
            std::swap(I[col], I[piv]);
        }
        cplx d = A[(size_t)col * N + col];
        if (std::abs(d) < 1e-300)
            continue;
        for (int r = col + 1; r < N; ++r)
        {
            cplx f = A[(size_t)r * N + col] / d;
            if (std::abs(f) == 0.0)
                continue;
            for (int j = col; j < N; ++j)
                A[(size_t)r * N + j] -= f * A[(size_t)col * N + j];
            I[r] -= f * I[col];
        }
    }
    for (int r = N - 1; r >= 0; --r)
    {
        cplx s = I[r];
        for (int j = r + 1; j < N; ++j)
            s -= A[(size_t)r * N + j] * I[j];
        cplx d = A[(size_t)r * N + r];
        I[r] = (std::abs(d) > 1e-300) ? s / d : cplx(0, 0);
    }
    resNorm = 0.0f;
}

//---------------------------------------------------------------------------
void MomSolver::postProcess()
{
    // node -> basis coefficient (0 at wire ends)
    std::vector<cplx> nodeI(nodes.size(), cplx(0, 0));
    for (int b = 0; b < N; ++b)
        nodeI[basis[b].node] = I[b];

    segMag.assign(segs.size(), 0.0f);
    for (size_t s = 0; s < segs.size(); ++s)
    {
        cplx im = 0.5 * (nodeI[segs[s].n0] + nodeI[segs[s].n1]);
        segMag[s] = (float)std::abs(im);
    }

    if (hasFeed && feedBasis >= 0 && std::abs(I[feedBasis]) > 1e-30)
    {
        zinVal = 1.0 / I[feedBasis];      // delta gap V = 1
        zinOk  = true;
    }
}

//---------------------------------------------------------------------------
void MomSolver::run()
{
    running = true;
    phaseText = "assembling";
    assemble();
    if (N == 0 || stopFlag)
    {
        phaseText = "no unknowns";
        finished = true; running = false;
        return;
    }
    maxStep = N;
    V.assign(N, cplx(0, 0));
    if (hasFeed && feedBasis >= 0)
        V[feedBasis] = cplx(1, 0);

    usedGpu = false;
    if (useGpu)
    {
        phaseText = "fill + solve (GPU)";
        usedGpu = RunGpuMom(*this, gpuMsg);
    }
    if (!usedGpu)
    {
        phaseText = "filling matrix (CPU)";
        fillMatrixCpu();
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
void MomSolver::getWireCurrents(std::vector<Vec3> &pts, std::vector<float> &mag)
{
    pts.clear();
    mag.clear();
    for (size_t s = 0; s < segs.size(); ++s)
    {
        pts.push_back(nodes[segs[s].n0]);
        pts.push_back(nodes[segs[s].n1]);
        mag.push_back(s < segMag.size() ? segMag[s] : 0.0f);
    }
}

//---------------------------------------------------------------------------
// radiation pattern from the line currents (segment Hertzian dipoles)
//---------------------------------------------------------------------------
bool MomSolver::computeFarField(FarFieldData &out, int nTheta, int nPhi)
{
    if (N == 0 || segs.empty())
        return false;
    const double k = 2.0 * M_PI * f0 / C0;
    std::vector<cplx> nodeI(nodes.size(), cplx(0, 0));
    for (int b = 0; b < N; ++b)
        nodeI[basis[b].node] = I[b];

    // per-segment complex moment = Iseg * len * tangent
    struct Mom { Vec3 mid, t; cplx I; float len; };
    std::vector<Mom> mom;
    mom.reserve(segs.size());
    for (const auto &s : segs)
    {
        cplx im = 0.5 * (nodeI[s.n0] + nodeI[s.n1]);
        if (std::abs(im) < 1e-30)
            continue;
        mom.push_back({ s.mid, s.t, im, s.len });
    }
    if (mom.empty())
        return false;

    out.nTheta = nTheta;
    out.nPhi   = nPhi;
    out.U.assign((size_t)nTheta * nPhi, 0.0f);
    for (int ti = 0; ti < nTheta; ++ti)
    {
        double th = M_PI * ti / (nTheta - 1);
        double st = std::sin(th), ct = std::cos(th);
        for (int pi = 0; pi < nPhi; ++pi)
        {
            double ph = 2.0 * M_PI * pi / (nPhi - 1);
            double cp = std::cos(ph), sp = std::sin(ph);
            Vec3 rh((float)(st*cp), (float)(st*sp), (float)ct);
            Vec3 thh((float)(ct*cp), (float)(ct*sp), (float)(-st));
            Vec3 phh((float)(-sp), (float)cp, 0.0f);
            cplx Nth(0, 0), Nph(0, 0);
            for (const auto &mm : mom)
            {
                double phase = k * (rh.x*mm.mid.x + rh.y*mm.mid.y + rh.z*mm.mid.z);
                cplx e = std::exp(cplx(0, phase)) * (double)mm.len;
                cplx c = mm.I * e;
                Nth += c * (double)(mm.t.x*thh.x + mm.t.y*thh.y + mm.t.z*thh.z);
                Nph += c * (double)(mm.t.x*phh.x + mm.t.y*phh.y + mm.t.z*phh.z);
            }
            out.U[(size_t)ti*nPhi + pi] =
                (float)(std::norm(Nth) + std::norm(Nph));
        }
    }
    float uMax = 0; int iMax = 0;
    double prad = 0;
    double dTh = M_PI / (nTheta - 1), dPh = 2.0*M_PI/(nPhi - 1);
    for (int ti = 0; ti < nTheta; ++ti)
    {
        double th = M_PI * ti / (nTheta - 1);
        double wTh = (ti == 0 || ti == nTheta-1) ? 0.5 : 1.0;
        for (int pi = 0; pi < nPhi; ++pi)
        {
            double wPh = (pi == 0 || pi == nPhi-1) ? 0.5 : 1.0;
            float u = out.U[(size_t)ti*nPhi + pi];
            if (u > uMax) { uMax = u; iMax = ti*nPhi + pi; }
            prad += (double)u * std::sin(th) * wTh * wPh * dTh * dPh;
        }
    }
    out.uMax = uMax;
    out.directivity = (prad > 0) ? (float)(4.0 * M_PI * uMax / prad) : 0.0f;
    out.peakThetaDeg = 180.0f * (iMax / nPhi) / (nTheta - 1);
    out.peakPhiDeg   = 360.0f * (iMax % nPhi) / (nPhi - 1);
    return true;
}
