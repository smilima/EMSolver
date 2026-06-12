//---------------------------------------------------------------------------
// FdtdSolver.cpp - Yee-grid FDTD implementation
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "FdtdSolver.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#pragma package(smart_init)

static const float C0   = 299792458.0f;
static const float EPS0 = 8.8541878128e-12f;
static const float MU0  = 1.25663706212e-6f;
static const float ETA0 = 376.730313f;

static const int POLT[3][2] = { {1, 2}, {0, 2}, {0, 1} };
static const int CYC_I[3] = { 1, 2, 0 };
static const int CYC_J[3] = { 2, 0, 1 };

//---------------------------------------------------------------------------
FdtdSolver::FdtdSolver()
{
    int n = (int)std::thread::hardware_concurrency();
    n = std::max(1, std::min(16, n - 1));
    pool = new ThreadPool(n);
    gpuMsg = "FDTD runs on the CPU (GPU port not implemented yet)";
}

//---------------------------------------------------------------------------
FdtdSolver::~FdtdSolver()
{
    delete pool;
}

//---------------------------------------------------------------------------
// cell-centered field averages
//---------------------------------------------------------------------------
float FdtdSolver::ecX(int i, int j, int k) const
{
    return 0.25f * (Ex[iEx(i, j, k)] + Ex[iEx(i, j + 1, k)] +
                    Ex[iEx(i, j, k + 1)] + Ex[iEx(i, j + 1, k + 1)]);
}

//---------------------------------------------------------------------------
float FdtdSolver::ecY(int i, int j, int k) const
{
    return 0.25f * (Ey[iEy(i, j, k)] + Ey[iEy(i + 1, j, k)] +
                    Ey[iEy(i, j, k + 1)] + Ey[iEy(i + 1, j, k + 1)]);
}

//---------------------------------------------------------------------------
float FdtdSolver::ecZ(int i, int j, int k) const
{
    return 0.25f * (Ez[iEz(i, j, k)] + Ez[iEz(i + 1, j, k)] +
                    Ez[iEz(i, j + 1, k)] + Ez[iEz(i + 1, j + 1, k)]);
}

//---------------------------------------------------------------------------
float FdtdSolver::hcX(int i, int j, int k) const
{
    return 0.5f * (Hx[iHx(i, j, k)] + Hx[iHx(i + 1, j, k)]);
}

//---------------------------------------------------------------------------
float FdtdSolver::hcY(int i, int j, int k) const
{
    return 0.5f * (Hy[iHy(i, j, k)] + Hy[iHy(i, j + 1, k)]);
}

//---------------------------------------------------------------------------
float FdtdSolver::hcZ(int i, int j, int k) const
{
    return 0.5f * (Hz[iHz(i, j, k)] + Hz[iHz(i, j, k + 1)]);
}

//---------------------------------------------------------------------------
void FdtdSolver::setup(const VoxelGridSpec &grid,
                       std::vector<uint8_t> materials,
                       std::vector<MatProps> table, const TlmConfig &cfgIn)
{
    g        = grid;
    config   = cfgIn;
    mat      = std::move(materials);
    matTable = std::move(table);
    dt   = 0.99f * g.dl / (C0 * std::sqrt(3.0f));
    cMur = (C0 * dt - g.dl) / (C0 * dt + g.dl);

    Ex.assign((size_t)g.nx * (g.ny + 1) * (g.nz + 1), 0.0f);
    Ey.assign((size_t)(g.nx + 1) * g.ny * (g.nz + 1), 0.0f);
    Ez.assign((size_t)(g.nx + 1) * (g.ny + 1) * g.nz, 0.0f);
    Hx.assign((size_t)(g.nx + 1) * g.ny * g.nz, 0.0f);
    Hy.assign((size_t)g.nx * (g.ny + 1) * g.nz, 0.0f);
    Hz.assign((size_t)g.nx * g.ny * (g.nz + 1), 0.0f);

    buildCoefficients();

    murX0a.assign(Ey.size() / (g.nx + 1), 0.0f);   // sized generously below
    // (proper sizes are set in applyMur's first use via assign)

    sources.clear();
    portList.clear();
    frames.clear();
    energySteps.clear();
    energyVals.clear();
    curStep    = 0;
    finished   = false;
    stopFlag   = false;
    dftSamples = 0;
    huySamples = 0;

    BuildPecSurfaceFaces(g, mat, MAT_PEC, surfFaces);
    buildHuygens();
    jsInstant.assign(surfFaces.size(), 0.0f);
    jsDftRe.assign(surfFaces.size() * 2, 0.0f);
    jsDftIm.assign(surfFaces.size() * 2, 0.0f);
    jsDftMag.clear();

    // Mur plane buffers: store previous-step tangential E at layers 0 and 1
    murX0a.assign(Ey.size(), 0.0f);   murX0b.assign(Ez.size(), 0.0f);
    murX1a.assign(Ey.size(), 0.0f);   murX1b.assign(Ez.size(), 0.0f);
    murY0a.assign(Ex.size(), 0.0f);   murY0b.assign(Ez.size(), 0.0f);
    murY1a.assign(Ex.size(), 0.0f);   murY1b.assign(Ez.size(), 0.0f);
    murZ0a.assign(Ex.size(), 0.0f);   murZ0b.assign(Ey.size(), 0.0f);
    murZ1a.assign(Ex.size(), 0.0f);   murZ1b.assign(Ey.size(), 0.0f);
}

//---------------------------------------------------------------------------
// per-edge material coefficients (average of the 4 adjacent cells)
//---------------------------------------------------------------------------
void FdtdSolver::buildCoefficients()
{
    haveDiel = false;
    for (uint8_t m : mat)
        if (m >= MAT_DIEL0)
        {
            haveDiel = true;
            break;
        }
    cb0 = dt / (EPS0 * g.dl);

    mEx.assign(Ex.size(), 1);
    mEy.assign(Ey.size(), 1);
    mEz.assign(Ez.size(), 1);
    if (haveDiel)
    {
        caEx.assign(Ex.size(), 1.0f);  cbEx.assign(Ex.size(), cb0);
        caEy.assign(Ey.size(), 1.0f);  cbEy.assign(Ey.size(), cb0);
        caEz.assign(Ez.size(), 1.0f);  cbEz.assign(Ez.size(), cb0);
    }

    auto cellMat = [&](int i, int j, int k, bool &pec, float &epsr,
                       float &sigma)
    {
        if (i < 0 || j < 0 || k < 0 || i >= g.nx || j >= g.ny || k >= g.nz)
            return;                       // outside: vacuum, ignore
        uint8_t m = mat[g.cellIndex(i, j, k)];
        if (m == MAT_PEC)
            pec = true;
        else if (m >= MAT_DIEL0 && m - MAT_DIEL0 < (int)matTable.size())
        {
            epsr  += matTable[m - MAT_DIEL0].epsr - 1.0f;
            sigma += matTable[m - MAT_DIEL0].sigma;
        }
    };
    // generic per-edge classification: edge along 'axis' with the 4
    // surrounding cells offset in the two transverse directions
    auto classify = [&](int axis, int i, int j, int k,
                        std::vector<uint8_t> &mask, std::vector<float> *ca,
                        std::vector<float> *cb, size_t idx)
    {
        int t1 = POLT[axis][0], t2 = POLT[axis][1];
        bool pec = false;
        float epsr = 1.0f, sigma = 0.0f;   // accumulated as average below
        float eAcc = 0.0f, sAcc = 0.0f;
        int   cnt = 0;
        for (int d1 = -1; d1 <= 0; ++d1)
            for (int d2 = -1; d2 <= 0; ++d2)
            {
                int co[3] = { i, j, k };
                co[t1] += d1;
                co[t2] += d2;
                if (co[0] < 0 || co[1] < 0 || co[2] < 0 ||
                    co[0] >= g.nx || co[1] >= g.ny || co[2] >= g.nz)
                    continue;
                bool p = false;
                float e1 = 1.0f, s1 = 0.0f;
                cellMat(co[0], co[1], co[2], p, e1, s1);
                if (p)
                    pec = true;
                eAcc += e1;
                sAcc += s1;
                ++cnt;
            }
        if (pec)
        {
            mask[idx] = 0;
            if (ca) { (*ca)[idx] = 0.0f; (*cb)[idx] = 0.0f; }
            return;
        }
        if (ca && cnt > 0)
        {
            epsr  = eAcc / cnt;
            sigma = sAcc / cnt;
            float eps  = EPS0 * std::max(1.0f, epsr);
            float lf   = sigma * dt / (2.0f * eps);
            (*ca)[idx] = (1.0f - lf) / (1.0f + lf);
            (*cb)[idx] = (dt / (eps * g.dl)) / (1.0f + lf);
        }
    };

    for (int k = 0; k <= g.nz; ++k)
        for (int j = 0; j <= g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                classify(0, i, j, k, mEx, haveDiel ? &caEx : nullptr,
                         haveDiel ? &cbEx : nullptr, iEx(i, j, k));
    for (int k = 0; k <= g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i <= g.nx; ++i)
                classify(1, i, j, k, mEy, haveDiel ? &caEy : nullptr,
                         haveDiel ? &cbEy : nullptr, iEy(i, j, k));
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j <= g.ny; ++j)
            for (int i = 0; i <= g.nx; ++i)
                classify(2, i, j, k, mEz, haveDiel ? &caEz : nullptr,
                         haveDiel ? &cbEz : nullptr, iEz(i, j, k));
}

//---------------------------------------------------------------------------
void FdtdSolver::buildHuygens()
{
    huyFaces.clear();
    huyDft.clear();
    int o = config.huyOffset;
    if (o < 2)
        return;
    const int nn[3] = { g.nx, g.ny, g.nz };
    if (2 * o + 2 >= nn[0] || 2 * o + 2 >= nn[1] || 2 * o + 2 >= nn[2])
        return;
    Vec3 domCenter = g.origin + Vec3(g.nx * g.dl, g.ny * g.dl, g.nz * g.dl) * 0.5f;
    auto addFace = [&](int i, int j, int k, int axis, int outSign)
    {
        size_t c = g.cellIndex(i, j, k);
        if (mat[c] != MAT_AIR)
            return;
        HuyFace f;
        f.cell    = (int)c;
        f.axis    = (int8_t)axis;
        f.outSign = (int8_t)outSign;
        f.pos     = g.cellCenter(i, j, k) - domCenter;
        huyFaces.push_back(f);
    };
    int x0 = o, x1 = nn[0] - 1 - o;
    int y0 = o, y1 = nn[1] - 1 - o;
    int z0 = o, z1 = nn[2] - 1 - o;
    for (int k = z0; k <= z1; ++k)
        for (int j = y0; j <= y1; ++j)
        {
            addFace(x0, j, k, 0, -1);
            addFace(x1, j, k, 0, +1);
        }
    for (int k = z0; k <= z1; ++k)
        for (int i = x0; i <= x1; ++i)
        {
            addFace(i, y0, k, 1, -1);
            addFace(i, y1, k, 1, +1);
        }
    for (int j = y0; j <= y1; ++j)
        for (int i = x0; i <= x1; ++i)
        {
            addFace(i, j, z0, 2, -1);
            addFace(i, j, z1, 2, +1);
        }
    huyDft.assign(huyFaces.size() * 8, 0.0f);
}

//---------------------------------------------------------------------------
int FdtdSolver::addPort(const std::vector<size_t> &cells, int polAxis,
                        float amp)
{
    TlmPort p;
    p.polAxis = polAxis;
    p.amp     = amp;
    for (size_t c : cells)
        if (c < mat.size() && mat[c] == MAT_AIR)
            p.cells.push_back(c);
    p.vRec.reserve(config.totalSteps);
    p.iRec.reserve(config.totalSteps);
    portList.push_back(std::move(p));
    return (int)portList.size() - 1;
}

//---------------------------------------------------------------------------
void FdtdSolver::addPlaneWave(int propAxis, int planeIndex, int polAxis,
                              float amp)
{
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                int co[3] = { i, j, k };
                if (co[propAxis] != planeIndex)
                    continue;
                size_t c = g.cellIndex(i, j, k);
                if (mat[c] == MAT_AIR)
                    sources.push_back({ c, polAxis, amp });
            }
}

//---------------------------------------------------------------------------
float FdtdSolver::waveformValue(int n) const
{
    float t = n * dt;
    float w = 2.0f * (float)M_PI * config.f0;
    switch (config.waveform)
    {
    case WaveformType::CwRamped:
    {
        float Tr = 3.0f / config.f0;
        float r  = (t < Tr) ? 0.5f * (1.0f - std::cos((float)M_PI * t / Tr)) : 1.0f;
        return r * std::sin(w * t);
    }
    case WaveformType::GaussianSine:
    {
        float tau = 2.0f / config.f0, t0 = 3.0f * tau;
        float u = (t - t0) / tau;
        return std::exp(-u * u) * std::sin(w * (t - t0));
    }
    case WaveformType::GaussianPulse:
    default:
    {
        float tau = 0.5f / config.f0, t0 = 3.0f * tau;
        float u = (t - t0) / tau;
        return std::exp(-u * u);
    }
    }
}

//---------------------------------------------------------------------------
void FdtdSolver::run()
{
    running = true;
    for (int n = 0; n < config.totalSteps && !stopFlag; ++n)
    {
        step(n);
        curStep = n + 1;
    }
    finalizeDft();
    finished = true;
    running  = false;
}

//---------------------------------------------------------------------------
void FdtdSolver::finalizeDft()
{
    std::lock_guard<std::mutex> lk(vizMutex);
    jsDftMag.assign(surfFaces.size(), 0.0f);
    if (dftSamples > 0)
    {
        float norm = 2.0f / dftSamples;
        for (size_t f = 0; f < surfFaces.size(); ++f)
        {
            float r1 = jsDftRe[f * 2],     i1 = jsDftIm[f * 2];
            float r2 = jsDftRe[f * 2 + 1], i2 = jsDftIm[f * 2 + 1];
            jsDftMag[f] = norm * std::sqrt(r1 * r1 + i1 * i1 +
                                           r2 * r2 + i2 * i2);
        }
    }
}

//---------------------------------------------------------------------------
void FdtdSolver::step(int n)
{
    // 1. soft sources: impress E along polAxis on the cell's 4 edges
    float sv = waveformValue(n);
    auto inject = [&](size_t cell, int polAxis, float amp)
    {
        int i = (int)(cell % g.nx);
        int j = (int)((cell / g.nx) % g.ny);
        int k = (int)(cell / ((size_t)g.nx * g.ny));
        float q = -0.25f * amp * sv;     // matches TLM sign convention
        if (polAxis == 0)
        {
            if (mEx[iEx(i, j, k)])         Ex[iEx(i, j, k)] += q;
            if (mEx[iEx(i, j + 1, k)])     Ex[iEx(i, j + 1, k)] += q;
            if (mEx[iEx(i, j, k + 1)])     Ex[iEx(i, j, k + 1)] += q;
            if (mEx[iEx(i, j + 1, k + 1)]) Ex[iEx(i, j + 1, k + 1)] += q;
        }
        else if (polAxis == 1)
        {
            if (mEy[iEy(i, j, k)])         Ey[iEy(i, j, k)] += q;
            if (mEy[iEy(i + 1, j, k)])     Ey[iEy(i + 1, j, k)] += q;
            if (mEy[iEy(i, j, k + 1)])     Ey[iEy(i, j, k + 1)] += q;
            if (mEy[iEy(i + 1, j, k + 1)]) Ey[iEy(i + 1, j, k + 1)] += q;
        }
        else
        {
            if (mEz[iEz(i, j, k)])         Ez[iEz(i, j, k)] += q;
            if (mEz[iEz(i + 1, j, k)])     Ez[iEz(i + 1, j, k)] += q;
            if (mEz[iEz(i, j + 1, k)])     Ez[iEz(i, j + 1, k)] += q;
            if (mEz[iEz(i + 1, j + 1, k)]) Ez[iEz(i + 1, j + 1, k)] += q;
        }
    };
    for (const auto &s : sources)
        inject(s.cell, s.polAxis, s.amp);
    for (const auto &p : portList)
        for (size_t c : p.cells)
            inject(c, p.polAxis, p.amp);

    // 2. monitors on current fields
    monitors(n);

    // 3. half-step H, then E with Mur boundaries (applyMur snapshots the
    //    boundary layers, runs updateE, then rewrites the boundary)
    updateH();
    applyMur();
}

//---------------------------------------------------------------------------
void FdtdSolver::updateH()
{
    const float ch = dt / (MU0 * g.dl);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    int chunks = pool->threadCount();
    pool->run(chunks, [&](int c)
    {
        int k0 = (int)((long long)nz * c / chunks);
        int k1 = (int)((long long)nz * (c + 1) / chunks);
        for (int k = k0; k < k1; ++k)
        {
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i <= nx; ++i)
                    Hx[iHx(i, j, k)] -= ch *
                        ((Ez[iEz(i, j + 1, k)] - Ez[iEz(i, j, k)]) -
                         (Ey[iEy(i, j, k + 1)] - Ey[iEy(i, j, k)]));
            for (int j = 0; j <= ny; ++j)
                for (int i = 0; i < nx; ++i)
                    Hy[iHy(i, j, k)] -= ch *
                        ((Ex[iEx(i, j, k + 1)] - Ex[iEx(i, j, k)]) -
                         (Ez[iEz(i + 1, j, k)] - Ez[iEz(i, j, k)]));
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    Hz[iHz(i, j, k)] -= ch *
                        ((Ey[iEy(i + 1, j, k)] - Ey[iEy(i, j, k)]) -
                         (Ex[iEx(i, j + 1, k)] - Ex[iEx(i, j, k)]));
        }
        // Hz top layer (k == nz)
        if (k1 == nz)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    Hz[iHz(i, j, nz)] -= ch *
                        ((Ey[iEy(i + 1, j, nz)] - Ey[iEy(i, j, nz)]) -
                         (Ex[iEx(i, j + 1, nz)] - Ex[iEx(i, j, nz)]));
    });
}

//---------------------------------------------------------------------------
void FdtdSolver::updateE()
{
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    int chunks = pool->threadCount();
    pool->run(chunks, [&](int c)
    {
        int k0 = (int)((long long)(nz + 1) * c / chunks);
        int k1 = (int)((long long)(nz + 1) * (c + 1) / chunks);
        for (int k = std::max(1, k0); k < k1 && k < nz; ++k)
        {
            // interior k for Ex, Ey (they need Hy/Hx at k-1)
            for (int j = 1; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    size_t q = iEx(i, j, k);
                    if (!mEx[q])
                    {
                        Ex[q] = 0.0f;
                        continue;
                    }
                    float curl = (Hz[iHz(i, j, k)] - Hz[iHz(i, j - 1, k)]) -
                                 (Hy[iHy(i, j, k)] - Hy[iHy(i, j, k - 1)]);
                    if (haveDiel)
                        Ex[q] = caEx[q] * Ex[q] + cbEx[q] * curl;
                    else
                        Ex[q] += cb0 * curl;
                }
            for (int j = 0; j < ny; ++j)
                for (int i = 1; i < nx; ++i)
                {
                    size_t q = iEy(i, j, k);
                    if (!mEy[q])
                    {
                        Ey[q] = 0.0f;
                        continue;
                    }
                    float curl = (Hx[iHx(i, j, k)] - Hx[iHx(i, j, k - 1)]) -
                                 (Hz[iHz(i, j, k)] - Hz[iHz(i - 1, j, k)]);
                    if (haveDiel)
                        Ey[q] = caEy[q] * Ey[q] + cbEy[q] * curl;
                    else
                        Ey[q] += cb0 * curl;
                }
        }
        // Ez: k in [k0, min(k1, nz)), interior i,j
        for (int k = k0; k < k1 && k < nz; ++k)
            for (int j = 1; j < ny; ++j)
                for (int i = 1; i < nx; ++i)
                {
                    size_t q = iEz(i, j, k);
                    if (!mEz[q])
                    {
                        Ez[q] = 0.0f;
                        continue;
                    }
                    float curl = (Hy[iHy(i, j, k)] - Hy[iHy(i - 1, j, k)]) -
                                 (Hx[iHx(i, j, k)] - Hx[iHx(i, j - 1, k)]);
                    if (haveDiel)
                        Ez[q] = caEz[q] * Ez[q] + cbEz[q] * curl;
                    else
                        Ez[q] += cb0 * curl;
                }
    });
}

//---------------------------------------------------------------------------
// first-order Mur: E_b^{n+1} = E_1^n + cMur (E_1^{n+1} - E_b^n)
// Implementation: snapshot layers 0/1 before the E update (called from
// step() just before updateE), then rewrite layer 0 afterwards. To keep it
// simple the snapshot+apply both live here, called around updateE().
//---------------------------------------------------------------------------
void FdtdSolver::applyMur()
{
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    // ---- snapshot pre-update values of layers 0 and 1 ----
    auto snapEy = [&](int i, std::vector<float> &dst)
    {
        for (int k = 0; k <= nz; ++k)
            for (int j = 0; j < ny; ++j)
                dst[iEy(i, j, k)] = Ey[iEy(i, j, k)];
    };
    auto snapEz = [&](int i, std::vector<float> &dst)
    {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j <= ny; ++j)
                dst[iEz(i, j, k)] = Ez[iEz(i, j, k)];
    };
    auto snapExY = [&](int j, std::vector<float> &dst)
    {
        for (int k = 0; k <= nz; ++k)
            for (int i = 0; i < nx; ++i)
                dst[iEx(i, j, k)] = Ex[iEx(i, j, k)];
    };
    auto snapEzY = [&](int j, std::vector<float> &dst)
    {
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i <= nx; ++i)
                dst[iEz(i, j, k)] = Ez[iEz(i, j, k)];
    };
    auto snapExZ = [&](int k, std::vector<float> &dst)
    {
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i)
                dst[iEx(i, j, k)] = Ex[iEx(i, j, k)];
    };
    auto snapEyZ = [&](int k, std::vector<float> &dst)
    {
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i)
                dst[iEy(i, j, k)] = Ey[iEy(i, j, k)];
    };

    // snapshots (old values at boundary layer and first interior layer are
    // stored interleaved in the same arrays at their natural indices)
    snapEy(0, murX0a);  snapEy(1, murX0a);
    snapEz(0, murX0b);  snapEz(1, murX0b);
    snapEy(nx, murX1a); snapEy(nx - 1, murX1a);
    snapEz(nx, murX1b); snapEz(nx - 1, murX1b);
    snapExY(0, murY0a);  snapExY(1, murY0a);
    snapEzY(0, murY0b);  snapEzY(1, murY0b);
    snapExY(ny, murY1a); snapExY(ny - 1, murY1a);
    snapEzY(ny, murY1b); snapEzY(ny - 1, murY1b);
    snapExZ(0, murZ0a);  snapExZ(1, murZ0a);
    snapEyZ(0, murZ0b);  snapEyZ(1, murZ0b);
    snapExZ(nz, murZ1a); snapExZ(nz - 1, murZ1a);
    snapEyZ(nz, murZ1b); snapEyZ(nz - 1, murZ1b);

    updateE();

    // ---- apply Mur on the 6 faces ----
    auto mur = [&](float &eb, float e1new, float e1old, float ebold)
    {
        eb = e1old + cMur * (e1new - ebold);
    };
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
        {
            mur(Ey[iEy(0, j, k)],  Ey[iEy(1, j, k)],
                murX0a[iEy(1, j, k)],  murX0a[iEy(0, j, k)]);
            mur(Ey[iEy(nx, j, k)], Ey[iEy(nx - 1, j, k)],
                murX1a[iEy(nx - 1, j, k)], murX1a[iEy(nx, j, k)]);
        }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
        {
            mur(Ez[iEz(0, j, k)],  Ez[iEz(1, j, k)],
                murX0b[iEz(1, j, k)],  murX0b[iEz(0, j, k)]);
            mur(Ez[iEz(nx, j, k)], Ez[iEz(nx - 1, j, k)],
                murX1b[iEz(nx - 1, j, k)], murX1b[iEz(nx, j, k)]);
        }
    for (int k = 0; k <= nz; ++k)
        for (int i = 0; i < nx; ++i)
        {
            mur(Ex[iEx(i, 0, k)],  Ex[iEx(i, 1, k)],
                murY0a[iEx(i, 1, k)],  murY0a[iEx(i, 0, k)]);
            mur(Ex[iEx(i, ny, k)], Ex[iEx(i, ny - 1, k)],
                murY1a[iEx(i, ny - 1, k)], murY1a[iEx(i, ny, k)]);
        }
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i <= nx; ++i)
        {
            mur(Ez[iEz(i, 0, k)],  Ez[iEz(i, 1, k)],
                murY0b[iEz(i, 1, k)],  murY0b[iEz(i, 0, k)]);
            mur(Ez[iEz(i, ny, k)], Ez[iEz(i, ny - 1, k)],
                murY1b[iEz(i, ny - 1, k)], murY1b[iEz(i, ny, k)]);
        }
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            mur(Ex[iEx(i, j, 0)],  Ex[iEx(i, j, 1)],
                murZ0a[iEx(i, j, 1)],  murZ0a[iEx(i, j, 0)]);
            mur(Ex[iEx(i, j, nz)], Ex[iEx(i, j, nz - 1)],
                murZ1a[iEx(i, j, nz - 1)], murZ1a[iEx(i, j, nz)]);
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i <= nx; ++i)
        {
            mur(Ey[iEy(i, j, 0)],  Ey[iEy(i, j, 1)],
                murZ0b[iEy(i, j, 1)],  murZ0b[iEy(i, j, 0)]);
            mur(Ey[iEy(i, j, nz)], Ey[iEy(i, j, nz - 1)],
                murZ1b[iEy(i, j, nz - 1)], murZ1b[iEy(i, j, nz)]);
        }
}

//---------------------------------------------------------------------------
void FdtdSolver::monitors(int n)
{
    const bool doDft = (n >= config.settleSteps);
    const float phase = 2.0f * (float)M_PI * config.f0 * (n * dt);
    const float cph = std::cos(phase), sph = std::sin(phase);
    const int nf = (int)surfFaces.size();

    auto cellCo = [&](int cell, int co[3])
    {
        co[0] = cell % g.nx;
        co[1] = (cell / g.nx) % g.ny;
        co[2] = cell / (g.nx * g.ny);
    };
    auto hAt = [&](int i, int j, int k, int comp) -> float
    {
        return comp == 0 ? hcX(i, j, k)
             : comp == 1 ? hcY(i, j, k) : hcZ(i, j, k);
    };
    auto eAt = [&](int i, int j, int k, int comp) -> float
    {
        return comp == 0 ? ecX(i, j, k)
             : comp == 1 ? ecY(i, j, k) : ecZ(i, j, k);
    };

    {
        std::lock_guard<std::mutex> lk(vizMutex);
        int chunks = pool->threadCount();

        // surface currents (eta0*H so magnitudes are E-scaled)
        pool->run(chunks, [&](int c)
        {
            int f0i = (int)((long long)nf * c / chunks);
            int f1i = (int)((long long)nf * (c + 1) / chunks);
            for (int f = f0i; f < f1i; ++f)
            {
                const SurfaceFace &sf = surfFaces[f];
                int co[3];
                cellCo(sf.airCell, co);
                int t1 = POLT[sf.axis][0], t2 = POLT[sf.axis][1];
                float h1 = ETA0 * hAt(co[0], co[1], co[2], t1);
                float h2 = ETA0 * hAt(co[0], co[1], co[2], t2);
                jsInstant[f] = std::sqrt(h1 * h1 + h2 * h2);
                if (doDft)
                {
                    jsDftRe[f * 2]     += h1 * cph;
                    jsDftIm[f * 2]     -= h1 * sph;
                    jsDftRe[f * 2 + 1] += h2 * cph;
                    jsDftIm[f * 2 + 1] -= h2 * sph;
                }
            }
        });
        if (doDft)
            ++dftSamples;

        // Huygens DFT (E and eta0*H, consistent pair for the NTFF)
        if (doDft && !huyFaces.empty())
        {
            const int nh = (int)huyFaces.size();
            pool->run(chunks, [&](int c)
            {
                int h0 = (int)((long long)nh * c / chunks);
                int h1i = (int)((long long)nh * (c + 1) / chunks);
                for (int f = h0; f < h1i; ++f)
                {
                    const HuyFace &hf = huyFaces[f];
                    int co[3];
                    cellCo(hf.cell, co);
                    int t1 = POLT[hf.axis][0], t2 = POLT[hf.axis][1];
                    float e1 = eAt(co[0], co[1], co[2], t1);
                    float e2 = eAt(co[0], co[1], co[2], t2);
                    float m1 = ETA0 * hAt(co[0], co[1], co[2], t1);
                    float m2 = ETA0 * hAt(co[0], co[1], co[2], t2);
                    float *acc = &huyDft[(size_t)f * 8];
                    acc[0] += e1 * cph;  acc[1] -= e1 * sph;
                    acc[2] += e2 * cph;  acc[3] -= e2 * sph;
                    acc[4] += m1 * cph;  acc[5] -= m1 * sph;
                    acc[6] += m2 * cph;  acc[7] -= m2 * sph;
                }
            });
            ++huySamples;
        }

        // port V/I records
        for (auto &p : portList)
        {
            float vSum = 0.0f;
            for (size_t c : p.cells)
            {
                int co[3];
                cellCo((int)c, co);
                vSum += -eAt(co[0], co[1], co[2], p.polAxis) * g.dl;
            }
            float iVal = 0.0f;
            if (!p.cells.empty())
            {
                size_t c = p.cells[p.cells.size() / 2];
                int co[3];
                cellCo((int)c, co);
                int a = p.polAxis;
                int t1 = CYC_I[a], t2 = CYC_J[a];
                int d1[3] = { 0, 0, 0 }, d2[3] = { 0, 0, 0 };
                d1[t1] = 1;
                d2[t2] = 1;
                // Ampere loop, square of side 2dl through the 4 neighbours
                iVal = 2.0f * g.dl * (
                    hAt(co[0] + d1[0], co[1] + d1[1], co[2] + d1[2], t2) -
                    hAt(co[0] - d1[0], co[1] - d1[1], co[2] - d1[2], t2) -
                    hAt(co[0] + d2[0], co[1] + d2[1], co[2] + d2[2], t1) +
                    hAt(co[0] - d2[0], co[1] - d2[1], co[2] - d2[2], t1));
            }
            p.vRec.push_back(vSum);
            p.iRec.push_back(iVal);
        }

        // field cut plane (|E|)
        int pa = planeAxis.load();
        if (pa >= 0 && (n % 2) == 0)
        {
            const int nn[3] = { g.nx, g.ny, g.nz };
            int a1 = (pa == 0) ? 1 : 0;
            int a2 = (pa == 2) ? 1 : 2;
            planeN1 = nn[a1];
            planeN2 = nn[a2];
            planeBuf.assign((size_t)planeN1 * planeN2, 0.0f);
            int idx = std::min(std::max(planeIndex, 0), nn[pa] - 1);
            for (int q2 = 0; q2 < planeN2; ++q2)
                for (int q1 = 0; q1 < planeN1; ++q1)
                {
                    int co[3];
                    co[pa] = idx;
                    co[a1] = q1;
                    co[a2] = q2;
                    float ex = eAt(co[0], co[1], co[2], 0);
                    float ey = eAt(co[0], co[1], co[2], 1);
                    float ez = eAt(co[0], co[1], co[2], 2);
                    planeBuf[(size_t)q2 * planeN1 + q1] =
                        std::sqrt(ex * ex + ey * ey + ez * ez) * g.dl;
                }
        }

        // playback frame recording
        if (config.recordEvery > 0 && (n % config.recordEvery) == 0)
        {
            VizFrame fr;
            fr.step = n;
            fr.js = jsInstant;
            int pax = planeAxis.load();
            if (pax >= 0 && !planeBuf.empty())
            {
                fr.plane     = planeBuf;
                fr.planeAxis = pax;
                fr.planeIdx  = planeIndex;
                fr.planeN1   = planeN1;
                fr.planeN2   = planeN2;
            }
            frames.push_back(std::move(fr));
        }
    }

    // total field energy ~ sum(E^2) + sum((eta0 H)^2), every 16 steps
    if ((n & 15) == 0)
    {
        int chunks = pool->threadCount();
        std::vector<double> part(chunks, 0.0);
        auto sumArr = [&](const std::vector<float> &a, double scale)
        {
            pool->run(chunks, [&](int c)
            {
                size_t lo = a.size() * c / chunks,
                       hi = a.size() * (c + 1) / chunks;
                double s = 0.0;
                for (size_t i = lo; i < hi; ++i)
                    s += (double)a[i] * a[i];
                part[c] += s * scale;
            });
        };
        for (auto &p : part)
            p = 0.0;
        sumArr(Ex, 1.0);
        sumArr(Ey, 1.0);
        sumArr(Ez, 1.0);
        double eta2 = (double)ETA0 * ETA0;
        sumArr(Hx, eta2);
        sumArr(Hy, eta2);
        sumArr(Hz, eta2);
        double e = 0.0;
        for (double p : part)
            e += p;
        energy = (float)e;
        {
            std::lock_guard<std::mutex> lk(vizMutex);
            energySteps.push_back(n);
            energyVals.push_back((float)e);
        }
    }
}

//---------------------------------------------------------------------------
// accessors (same semantics as TlmSolver)
//---------------------------------------------------------------------------
void FdtdSolver::getJsInstant(std::vector<float> &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    out = jsInstant;
}

//---------------------------------------------------------------------------
bool FdtdSolver::getJsDft(std::vector<float> &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    if (jsDftMag.empty())
        return false;
    out = jsDftMag;
    return true;
}

//---------------------------------------------------------------------------
void FdtdSolver::setFieldPlane(int axis, int index)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    planeIndex = index;
    planeAxis  = axis;
}

//---------------------------------------------------------------------------
bool FdtdSolver::getFieldPlane(std::vector<float> &out, int &axis, int &index,
                               int &n1, int &n2)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    if (planeAxis < 0 || planeBuf.empty())
        return false;
    out   = planeBuf;
    axis  = planeAxis;
    index = planeIndex;
    n1    = planeN1;
    n2    = planeN2;
    return true;
}

//---------------------------------------------------------------------------
bool FdtdSolver::computeFarField(FarFieldData &out, int nTheta, int nPhi)
{
    return ComputeFarFieldFromHuygens(huyFaces, huyDft, huySamples,
                                      config.f0, pool, out, nTheta, nPhi);
}

//---------------------------------------------------------------------------
int FdtdSolver::frameCount()
{
    std::lock_guard<std::mutex> lk(vizMutex);
    return (int)frames.size();
}

//---------------------------------------------------------------------------
bool FdtdSolver::getFrame(int idx, VizFrame &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    if (idx < 0 || idx >= (int)frames.size())
        return false;
    out = frames[idx];
    return true;
}

//---------------------------------------------------------------------------
void FdtdSolver::getEnergyHistory(std::vector<int> &steps,
                                  std::vector<float> &vals)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    steps = energySteps;
    vals  = energyVals;
}
