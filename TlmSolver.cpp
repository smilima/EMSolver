//---------------------------------------------------------------------------
// TlmSolver.cpp - SCN TLM solver implementation
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "TlmSolver.h"
#include "GpuTlm.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#pragma package(smart_init)

static const float C0  = 299792458.0f;
static const float ETA0 = 376.730313f;

//---------------------------------------------------------------------------
// Port conventions
//   port = axis*4 + side*2 + pol,  side 0 = '-' face, 1 = '+' face
//   pol selects the E-field axis: POL[axis][pol]
//---------------------------------------------------------------------------
static const int POL[3][2] = { {1, 2}, {0, 2}, {0, 1} };

static inline int portOf(int axis, int side, int polAxis)
{
    int pol = (polAxis == POL[axis][0]) ? 0 : 1;
    return axis * 4 + side * 2 + pol;
}

//---------------------------------------------------------------------------
// cyclic pair (i,j) for circulation axis k: e_i x e_j = e_k
static const int CYC_I[3] = { 1, 2, 0 };
static const int CYC_J[3] = { 2, 0, 1 };

// link-line circulation about axis k at a cell (from incident pulses)
static inline float ZIof(const float *vp, int k)
{
    int i = CYC_I[k], j = CYC_J[k];
    return 0.5f * (vp[portOf(i, 0, j)] - vp[portOf(i, 1, j)] -
                   vp[portOf(j, 0, i)] + vp[portOf(j, 1, i)]);
}

//---------------------------------------------------------------------------
// node common voltage for polarization j (air cell)
static inline float VjOf(const float *vp, int j)
{
    int a1 = (j == 0) ? 1 : 0;
    int a2 = (j == 2) ? 1 : 2;
    return 0.5f * (vp[portOf(a1, 0, j)] + vp[portOf(a1, 1, j)] +
                   vp[portOf(a2, 0, j)] + vp[portOf(a2, 1, j)]);
}

//---------------------------------------------------------------------------
// Scatter one SCN node: vin[12] incident -> vout[12] reflected.
// Field form:  Vr(i,s,j) = Vj + sgn * ZIk - Vin(i,1-s,j)
// Reproduces Johns' unitary 12x12 SCN scatter; see ScatterUnitarityError().
//---------------------------------------------------------------------------
static inline void ScatterCell(const float *vin, float *vout)
{
    float Vp[3], ZI[3];
    for (int j = 0; j < 3; ++j)
        Vp[j] = VjOf(vin, j);
    for (int k = 0; k < 3; ++k)
        ZI[k] = ZIof(vin, k);
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 2; ++s)
            for (int pol = 0; pol < 2; ++pol)
            {
                int j = POL[i][pol];
                int k = 3 - i - j;
                float sg = (((i + 1) % 3) == j ? 1.0f : -1.0f) * (s ? 1.0f : -1.0f);
                vout[i * 4 + s * 2 + pol] =
                    Vp[j] + sg * ZI[k] - vin[portOf(i, 1 - s, j)];
            }
}

//---------------------------------------------------------------------------
// Stub-loaded SCN (Christopoulos): 3 open-circuit capacitive stubs model
// excess permittivity, a shunt conductance models loss.
//   yh = 4*(epsr - 1)      (normalized stub admittance, dt = dl/2c)
//   gh = sigma * dl * eta0 (normalized conductance)
//   Vj = 2*(sum of 4 link incidents + yh*Vstub_j) / (4 + yh + gh)
// Link reflection keeps the same field form; stub reflects Vj - Vstub.
//---------------------------------------------------------------------------
static inline void ScatterCellDiel(const float *vin, float *vs,
                                   float yh, float gh, float *vout)
{
    float Vp[3], ZI[3];
    float den = 4.0f + yh + gh;
    for (int j = 0; j < 3; ++j)
    {
        int a1 = (j == 0) ? 1 : 0;
        int a2 = (j == 2) ? 1 : 2;
        float sum4 = vin[portOf(a1, 0, j)] + vin[portOf(a1, 1, j)] +
                     vin[portOf(a2, 0, j)] + vin[portOf(a2, 1, j)];
        Vp[j] = 2.0f * (sum4 + yh * vs[j]) / den;
    }
    for (int k = 0; k < 3; ++k)
        ZI[k] = ZIof(vin, k);
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 2; ++s)
            for (int pol = 0; pol < 2; ++pol)
            {
                int j = POL[i][pol];
                int k = 3 - i - j;
                float sg = (((i + 1) % 3) == j ? 1.0f : -1.0f) * (s ? 1.0f : -1.0f);
                vout[i * 4 + s * 2 + pol] =
                    Vp[j] + sg * ZI[k] - vin[portOf(i, 1 - s, j)];
            }
    for (int j = 0; j < 3; ++j)
        vs[j] = Vp[j] - vs[j];      // open stub: reflected returns next step
}

//---------------------------------------------------------------------------
float TlmSolver::ScatterUnitarityError()
{
    float S[12][12];
    for (int c = 0; c < 12; ++c)
    {
        float vin[12] = {0}, vout[12];
        vin[c] = 1.0f;
        ScatterCell(vin, vout);
        for (int r = 0; r < 12; ++r)
            S[r][c] = vout[r];
    }
    float maxErr = 0.0f;
    for (int a = 0; a < 12; ++a)
        for (int b = 0; b < 12; ++b)
        {
            float dot = 0.0f;
            for (int r = 0; r < 12; ++r)
                dot += S[r][a] * S[r][b];
            float want = (a == b) ? 1.0f : 0.0f;
            maxErr = std::max(maxErr, std::fabs(dot - want));
        }
    return maxErr;
}

//---------------------------------------------------------------------------
float TlmSolver::DielScatterUnitarityError(float epsr)
{
    // weighted unitarity: S^T W S = W, link weight 1, stub weight yh
    const int NP = 15;
    float yh = 4.0f * (epsr - 1.0f);
    float W[NP];
    for (int i = 0; i < 12; ++i) W[i] = 1.0f;
    for (int i = 12; i < 15; ++i) W[i] = yh;
    float S[NP][NP];
    for (int c = 0; c < NP; ++c)
    {
        float vin[12] = {0}, vs[3] = {0}, vout[12];
        if (c < 12) vin[c] = 1.0f; else vs[c - 12] = 1.0f;
        ScatterCellDiel(vin, vs, yh, 0.0f, vout);
        for (int r = 0; r < 12; ++r) S[r][c] = vout[r];
        for (int r = 0; r < 3; ++r)  S[12 + r][c] = vs[r];
    }
    float maxErr = 0.0f;
    for (int a = 0; a < NP; ++a)
        for (int b = 0; b < NP; ++b)
        {
            float dot = 0.0f;
            for (int r = 0; r < NP; ++r)
                dot += W[r] * S[r][a] * S[r][b];
            float want = (a == b) ? W[a] : 0.0f;
            maxErr = std::max(maxErr, std::fabs(dot - want));
        }
    return maxErr;
}

//---------------------------------------------------------------------------
// ThreadPool
//---------------------------------------------------------------------------
ThreadPool::ThreadPool(int numThreads)
{
    if (numThreads < 1) numThreads = 1;
    for (int i = 0; i < numThreads; ++i)
        workers.emplace_back([this] { workerMain(); });
}

//---------------------------------------------------------------------------
ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lk(m);
        quit = true;
        ++generation;
    }
    cvWork.notify_all();
    for (auto &w : workers)
        w.join();
}

//---------------------------------------------------------------------------
void ThreadPool::workerMain()
{
    uint64_t localGen = 0;
    for (;;)
    {
        {
            std::unique_lock<std::mutex> lk(m);
            cvWork.wait(lk, [&] { return quit || generation != localGen; });
            if (quit)
                return;
            localGen = generation;
        }
        for (;;)
        {
            int c = nextChunk.fetch_add(1);
            if (c >= chunksTotal)
                break;
            (*job)(c);
        }
        {
            std::lock_guard<std::mutex> lk(m);
            if (++doneWorkers == (int)workers.size())
                cvDone.notify_all();
        }
    }
}

//---------------------------------------------------------------------------
void ThreadPool::run(int chunks, const std::function<void(int)> &fn)
{
    if (chunks <= 0)
        return;
    {
        std::lock_guard<std::mutex> lk(m);
        job         = &fn;
        chunksTotal = chunks;
        nextChunk   = 0;
        doneWorkers = 0;
        ++generation;
    }
    cvWork.notify_all();
    {
        std::unique_lock<std::mutex> lk(m);
        cvDone.wait(lk, [&] { return doneWorkers == (int)workers.size(); });
    }
}

//---------------------------------------------------------------------------
// TlmSolver
//---------------------------------------------------------------------------
TlmSolver::TlmSolver()
{
    int n = (int)std::thread::hardware_concurrency();
    n = std::max(1, std::min(16, n - 1));
    pool = new ThreadPool(n);
}

//---------------------------------------------------------------------------
TlmSolver::~TlmSolver()
{
    delete pool;
}

//---------------------------------------------------------------------------
void TlmSolver::setup(const VoxelGridSpec &grid, std::vector<uint8_t> materials,
                      std::vector<MatProps> table, const TlmConfig &cfg)
{
    g        = grid;
    config   = cfg;
    mat      = std::move(materials);
    matTable = std::move(table);
    size_t n = (size_t)g.nx * g.ny * g.nz;
    V.assign(n * 12, 0.0f);
    dt = g.dl / (2.0f * C0);

    yhat.clear();
    ghat.clear();
    for (const auto &mp : matTable)
    {
        yhat.push_back(4.0f * (std::max(1.0f, mp.epsr) - 1.0f));
        ghat.push_back(std::max(0.0f, mp.sigma) * g.dl * ETA0);
    }
    bool anyDiel = false;
    for (uint8_t mm : mat)
        if (mm >= MAT_DIEL0) { anyDiel = true; break; }
    if (anyDiel)
        Vs.assign(n * 3, 0.0f);
    else
        Vs.clear();

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
    usedGpu    = false;
    gpuMsg.clear();
    buildSurfaceFaces();
    buildHuygens();
    jsInstant.assign(surfFaces.size(), 0.0f);
    jsDftRe.assign(surfFaces.size() * 2, 0.0f);
    jsDftIm.assign(surfFaces.size() * 2, 0.0f);
    jsDftMag.clear();
}

//---------------------------------------------------------------------------
void TlmSolver::buildSurfaceFaces()
{
    BuildPecSurfaceFaces(g, mat, MAT_PEC, surfFaces);
}

//---------------------------------------------------------------------------
void TlmSolver::buildHuygens()
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
int TlmSolver::addPort(const std::vector<size_t> &cells, int polAxis, float amp)
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
void TlmSolver::addPlaneWave(int propAxis, int planeIndex, int polAxis, float amp)
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
float TlmSolver::waveformValue(int n) const
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
void TlmSolver::run()
{
    running = true;
    usedGpu = false;
    if (config.useGpu)
        usedGpu = RunGpuTlm(*this, gpuMsg);
    if (!usedGpu)
        runCpu();
    finalizeDft();
    finished = true;
    running  = false;
}

//---------------------------------------------------------------------------
void TlmSolver::runCpu()
{
    for (int n = curStep; n < config.totalSteps && !stopFlag; ++n)
    {
        step(n);
        curStep = n + 1;
    }
}

//---------------------------------------------------------------------------
void TlmSolver::finalizeDft()
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
void TlmSolver::step(int n)
{
    // 1. source injection (soft sources add to incident pulses)
    float sv = waveformValue(n);
    auto inject = [&](size_t cell, int polAxis, float amp)
    {
        float q = 0.5f * amp * sv;
        float *vp = &V[portBase(cell)];
        for (int axis = 0; axis < 3; ++axis)
        {
            if (axis == polAxis)
                continue;
            vp[portOf(axis, 0, polAxis)] += q;
            vp[portOf(axis, 1, polAxis)] += q;
        }
    };
    for (const auto &s : sources)
        inject(s.cell, s.polAxis, s.amp);
    for (const auto &p : portList)
        for (size_t c : p.cells)
            inject(c, p.polAxis, p.amp);

    // 2. monitors on incident pulses
    monitors(n);

    // 3+4. scatter then connect
    scatter();
    connect();
}

//---------------------------------------------------------------------------
void TlmSolver::scatter()
{
    int chunks = pool->threadCount();
    int nz = g.nz;
    bool haveStubs = !Vs.empty();
    pool->run(chunks, [&](int c)
    {
        int k0 = (int)((long long)nz * c / chunks);
        int k1 = (int)((long long)nz * (c + 1) / chunks);
        size_t cellsPerSlab = (size_t)g.nx * g.ny;
        for (int k = k0; k < k1; ++k)
        {
            size_t cell = cellsPerSlab * k;
            for (size_t q = 0; q < cellsPerSlab; ++q, ++cell)
            {
                uint8_t mm = mat[cell];
                if (mm == MAT_PEC)
                    continue;
                float vin[12], vout[12];
                std::memcpy(vin, &V[cell * 12], sizeof(vin));
                if (mm >= MAT_DIEL0 && haveStubs)
                {
                    int id = mm - MAT_DIEL0;
                    ScatterCellDiel(vin, &Vs[cell * 3],
                                    yhat[id], ghat[id], vout);
                }
                else
                    ScatterCell(vin, vout);
                std::memcpy(&V[cell * 12], vout, sizeof(vout));
            }
        }
    });
}

//---------------------------------------------------------------------------
void TlmSolver::connect()
{
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const size_t sx = 1, sy = (size_t)nx, sz = (size_t)nx * ny;
    const float rho = config.boundaryRho;
    int chunks = pool->threadCount();

    pool->run(chunks, [&](int ch)
    {
        int k0 = (int)((long long)nz * ch / chunks);
        int k1 = (int)((long long)nz * (ch + 1) / chunks);

        auto link = [&](size_t c, size_t nb, int axis)
        {
            bool ca = (mat[c] != MAT_PEC), na = (mat[nb] != MAT_PEC);
            float *vc = &V[c * 12 + (size_t)axis * 4];
            float *vn = &V[nb * 12 + (size_t)axis * 4];
            for (int pol = 0; pol < 2; ++pol)
            {
                if (ca && na)
                    std::swap(vc[2 + pol], vn[pol]);
                else if (ca)            // neighbour is PEC: short circuit
                    vc[2 + pol] = -vc[2 + pol];
                else if (na)
                    vn[pol] = -vn[pol];
            }
        };
        auto outerFace = [&](size_t c, int axis, int side)
        {
            float *vp = &V[c * 12 + (size_t)axis * 4 + (size_t)side * 2];
            vp[0] *= rho;
            vp[1] *= rho;
        };

        for (int k = k0; k < k1; ++k)
            for (int j = 0; j < ny; ++j)
            {
                size_t row = sz * k + sy * j;
                for (int i = 0; i < nx - 1; ++i)
                    link(row + i, row + i + sx, 0);
                outerFace(row, 0, 0);
                outerFace(row + (nx - 1), 0, 1);
                if (j < ny - 1)
                    for (int i = 0; i < nx; ++i)
                        link(row + i, row + i + sy, 1);
                else
                    for (int i = 0; i < nx; ++i)
                        outerFace(row + i, 1, 1);
                if (j == 0)
                    for (int i = 0; i < nx; ++i)
                        outerFace(row + i, 1, 0);
                if (k < nz - 1)
                    for (int i = 0; i < nx; ++i)
                        link(row + i, row + i + sz, 2);
                else
                    for (int i = 0; i < nx; ++i)
                        outerFace(row + i, 2, 1);
                if (k == 0)
                    for (int i = 0; i < nx; ++i)
                        outerFace(row + i, 2, 0);
            }
    });
}

//---------------------------------------------------------------------------
void TlmSolver::monitors(int n)
{
    const bool doDft = (n >= config.settleSteps);
    const float phase = 2.0f * (float)M_PI * config.f0 * (n * dt);
    const float cph = std::cos(phase), sph = std::sin(phase);
    const int nf = (int)surfFaces.size();

    {
        std::lock_guard<std::mutex> lk(vizMutex);
        int chunks = pool->threadCount();

        // ---- surface currents on PEC faces ----
        pool->run(chunks, [&](int c)
        {
            int f0i = (int)((long long)nf * c / chunks);
            int f1i = (int)((long long)nf * (c + 1) / chunks);
            for (int f = f0i; f < f1i; ++f)
            {
                const SurfaceFace &sf = surfFaces[f];
                const float *vp = &V[(size_t)sf.airCell * 12];
                int t1 = POL[sf.axis][0], t2 = POL[sf.axis][1];
                float h1 = ZIof(vp, t1), h2 = ZIof(vp, t2);
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

        // ---- Huygens surface DFT (E and eta0*H tangential phasors) ----
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
                    const float *vp = &V[(size_t)hf.cell * 12];
                    int t1 = POL[hf.axis][0], t2 = POL[hf.axis][1];
                    // E_t ~ -V_t ; eta0*H_t ~ -ZI_t  (common 1/dl dropped)
                    float e1 = -VjOf(vp, t1), e2 = -VjOf(vp, t2);
                    float m1 = -ZIof(vp, t1), m2 = -ZIof(vp, t2);
                    float *acc = &huyDft[(size_t)f * 8];
                    acc[0] += e1 * cph;  acc[1] -= e1 * sph;
                    acc[2] += e2 * cph;  acc[3] -= e2 * sph;
                    acc[4] += m1 * cph;  acc[5] -= m1 * sph;
                    acc[6] += m2 * cph;  acc[7] -= m2 * sph;
                }
            });
            ++huySamples;
        }

        // ---- port V/I records (every step) ----
        const size_t strideOf[3] = { 1, (size_t)g.nx, (size_t)g.nx * g.ny };
        for (auto &p : portList)
        {
            float vSum = 0.0f;
            for (size_t c : p.cells)
                vSum += VjOf(&V[c * 12], p.polAxis);
            float iVal = 0.0f;
            if (!p.cells.empty())
            {
                size_t c = p.cells[p.cells.size() / 2];
                int a = p.polAxis;
                int t1 = CYC_I[a], t2 = CYC_J[a];   // t1 x t2 = a
                size_t s1 = strideOf[t1], s2 = strideOf[t2];
                // Ampere loop, square of side 2dl through the 4 transverse
                // neighbours; each side contributes H_mid * 2dl:
                // I = -(2/Z0) [ ZI_t2(+t1) - ZI_t2(-t1) - ZI_t1(+t2) + ZI_t1(-t2) ]
                float zi = 0.0f;
                zi += ZIof(&V[(c + s1) * 12], t2);
                zi -= ZIof(&V[(c - s1) * 12], t2);
                zi -= ZIof(&V[(c + s2) * 12], t1);
                zi += ZIof(&V[(c - s2) * 12], t1);
                iVal = -2.0f * zi / ETA0;
            }
            p.vRec.push_back(vSum);
            p.iRec.push_back(iVal);
        }

        // ---- field cut plane (|E|, arbitrary units) ----
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
                    size_t cell = g.cellIndex(co[0], co[1], co[2]);
                    if (mat[cell] == MAT_PEC)
                        continue;
                    const float *vp = &V[cell * 12];
                    float e2acc = 0.0f;
                    for (int j = 0; j < 3; ++j)
                    {
                        float vj = VjOf(vp, j);
                        e2acc += vj * vj;
                    }
                    planeBuf[(size_t)q2 * planeN1 + q1] = std::sqrt(e2acc);
                }
        }

        // ---- playback frame recording ----
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

    // energy monitor: total link-line energy sum(V^2), every 16 steps
    if ((n & 15) == 0)
    {
        int chunks = pool->threadCount();
        std::vector<double> part(chunks, 0.0);
        const size_t total = V.size();
        pool->run(chunks, [&](int c)
        {
            size_t a = total * c / chunks, b = total * (c + 1) / chunks;
            double s = 0.0;
            for (size_t i = a; i < b; ++i)
                s += (double)V[i] * V[i];
            part[c] = s;
        });
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
void TlmSolver::getEnergyHistory(std::vector<int> &steps,
                                 std::vector<float> &vals)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    steps = energySteps;
    vals  = energyVals;
}

//---------------------------------------------------------------------------
// Near-to-far-field transform (Huygens box, single frequency f0).
// Shared by the TLM and FDTD solvers.
//---------------------------------------------------------------------------
bool ComputeFarFieldFromHuygens(const std::vector<HuyFace> &huyFaces,
                                const std::vector<float> &huyDft,
                                int huySamples, float f0, ThreadPool *pool,
                                FarFieldData &out, int nTheta, int nPhi)
{
    if (huyFaces.empty() || huySamples == 0)
        return false;

    const float k0 = 2.0f * (float)M_PI * f0 / C0;
    const int nh = (int)huyFaces.size();
    out.nTheta = nTheta;
    out.nPhi   = nPhi;
    out.U.assign((size_t)nTheta * nPhi, 0.0f);

    // precompute per-face equivalent currents in world components (complex):
    //   J' = n x (eta0*H), M = -(n x E); e_a x e_t1 = sa*e_t2, e_a x e_t2 = -sa*e_t1
    struct FaceCur { Vec3 pos; float jRe[3], jIm[3], mRe[3], mIm[3]; };
    std::vector<FaceCur> fc(nh);
    for (int f = 0; f < nh; ++f)
    {
        const HuyFace &hf = huyFaces[f];
        const float *a = &huyDft[(size_t)f * 8];
        int ax = hf.axis;
        int t1 = POL[ax][0], t2 = POL[ax][1];
        float sa = (ax == 1) ? -1.0f : 1.0f;
        float s  = (float)hf.outSign;
        FaceCur &q = fc[f];
        q.pos = hf.pos;
        for (int c = 0; c < 3; ++c)
        {
            q.jRe[c] = q.jIm[c] = q.mRe[c] = q.mIm[c] = 0.0f;
        }
        // J' = s*sa*( H1*e_t2 - H2*e_t1 ),  H stored eta0-scaled in a[4..7]
        q.jRe[t2] =  s * sa * a[4];  q.jIm[t2] =  s * sa * a[5];
        q.jRe[t1] = -s * sa * a[6];  q.jIm[t1] = -s * sa * a[7];
        // M = -s*sa*( E1*e_t2 - E2*e_t1 ),  E stored in a[0..3]
        q.mRe[t2] = -s * sa * a[0];  q.mIm[t2] = -s * sa * a[1];
        q.mRe[t1] =  s * sa * a[2];  q.mIm[t1] =  s * sa * a[3];
    }

    int chunks = pool->threadCount();
    pool->run(chunks, [&](int ch)
    {
        int tA = (int)((long long)nTheta * ch / chunks);
        int tB = (int)((long long)nTheta * (ch + 1) / chunks);
        for (int ti = tA; ti < tB; ++ti)
        {
            float th = (float)M_PI * ti / (nTheta - 1);
            float st = std::sin(th), ct = std::cos(th);
            for (int pi = 0; pi < nPhi; ++pi)
            {
                float ph = 2.0f * (float)M_PI * pi / (nPhi - 1);
                float cp = std::cos(ph), sp = std::sin(ph);
                Vec3 rhat(st * cp, st * sp, ct);
                Vec3 that(ct * cp, ct * sp, -st);
                Vec3 phat(-sp, cp, 0.0f);

                float NtR = 0, NtI = 0, NpR = 0, NpI = 0;
                float LtR = 0, LtI = 0, LpR = 0, LpI = 0;
                for (int f = 0; f < nh; ++f)
                {
                    const FaceCur &q = fc[f];
                    float pha = k0 * (rhat.x * q.pos.x + rhat.y * q.pos.y +
                                      rhat.z * q.pos.z);
                    float c = std::cos(pha), s = std::sin(pha);
                    // project currents on theta/phi unit vectors
                    float jtR = q.jRe[0]*that.x + q.jRe[1]*that.y + q.jRe[2]*that.z;
                    float jtI = q.jIm[0]*that.x + q.jIm[1]*that.y + q.jIm[2]*that.z;
                    float jpR = q.jRe[0]*phat.x + q.jRe[1]*phat.y;
                    float jpI = q.jIm[0]*phat.x + q.jIm[1]*phat.y;
                    float mtR = q.mRe[0]*that.x + q.mRe[1]*that.y + q.mRe[2]*that.z;
                    float mtI = q.mIm[0]*that.x + q.mIm[1]*that.y + q.mIm[2]*that.z;
                    float mpR = q.mRe[0]*phat.x + q.mRe[1]*phat.y;
                    float mpI = q.mIm[0]*phat.x + q.mIm[1]*phat.y;
                    // multiply by e^{+j pha}
                    NtR += jtR * c - jtI * s;  NtI += jtR * s + jtI * c;
                    NpR += jpR * c - jpI * s;  NpI += jpR * s + jpI * c;
                    LtR += mtR * c - mtI * s;  LtI += mtR * s + mtI * c;
                    LpR += mpR * c - mpI * s;  LpI += mpR * s + mpI * c;
                }
                // U ~ |L_phi + N'_theta|^2 + |L_theta - N'_phi|^2
                float aR = LpR + NtR, aI = LpI + NtI;
                float bR = LtR - NpR, bI = LtI - NpI;
                out.U[(size_t)ti * nPhi + pi] = aR*aR + aI*aI + bR*bR + bI*bI;
            }
        }
    });

    // directivity
    float uMax = 0.0f;
    int iMax = 0;
    double prad = 0.0;
    float dTh = (float)M_PI / (nTheta - 1);
    float dPh = 2.0f * (float)M_PI / (nPhi - 1);
    for (int ti = 0; ti < nTheta; ++ti)
    {
        float th = (float)M_PI * ti / (nTheta - 1);
        float wTh = (ti == 0 || ti == nTheta - 1) ? 0.5f : 1.0f;
        for (int pi = 0; pi < nPhi; ++pi)
        {
            float wPh = (pi == 0 || pi == nPhi - 1) ? 0.5f : 1.0f;
            float u = out.U[(size_t)ti * nPhi + pi];
            if (u > uMax) { uMax = u; iMax = ti * nPhi + pi; }
            prad += (double)u * std::sin(th) * wTh * wPh * dTh * dPh;
        }
    }
    out.uMax = uMax;
    out.directivity = (prad > 0) ? (float)(4.0 * M_PI * uMax / prad) : 0.0f;
    out.peakThetaDeg = 180.0f * (iMax / nPhi) / (nTheta - 1);
    out.peakPhiDeg   = 360.0f * (iMax % nPhi) / (nPhi - 1);
    return true;
}

//---------------------------------------------------------------------------
bool TlmSolver::computeFarField(FarFieldData &out, int nTheta, int nPhi)
{
    return ComputeFarFieldFromHuygens(huyFaces, huyDft, huySamples,
                                      config.f0, pool, out, nTheta, nPhi);
}

//---------------------------------------------------------------------------
void TlmSolver::getJsInstant(std::vector<float> &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    out = jsInstant;
}

//---------------------------------------------------------------------------
int TlmSolver::frameCount()
{
    std::lock_guard<std::mutex> lk(vizMutex);
    return (int)frames.size();
}

//---------------------------------------------------------------------------
bool TlmSolver::getFrame(int idx, VizFrame &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    if (idx < 0 || idx >= (int)frames.size())
        return false;
    out = frames[idx];
    return true;
}

//---------------------------------------------------------------------------
bool TlmSolver::getJsDft(std::vector<float> &out)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    if (jsDftMag.empty())
        return false;
    out = jsDftMag;
    return true;
}

//---------------------------------------------------------------------------
void TlmSolver::setFieldPlane(int axis, int index)
{
    std::lock_guard<std::mutex> lk(vizMutex);
    planeIndex = index;
    planeAxis  = axis;
}

//---------------------------------------------------------------------------
bool TlmSolver::getFieldPlane(std::vector<float> &out, int &axis, int &index,
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
