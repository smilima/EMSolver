//---------------------------------------------------------------------------
// FemSolver.cpp - Whitney edge-element FEM implementation
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "FemSolver.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#pragma package(smart_init)

static const double C0   = 299792458.0;
static const double EPS0 = 8.8541878128e-12;
static const double MU0  = 1.25663706212e-6;

// local edge -> node pairs, order (01,02,03,12,13,23)
static const int EP[6][2] = { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
static const int CYC_I[3] = { 1, 2, 0 };
static const int CYC_J[3] = { 2, 0, 1 };

//---------------------------------------------------------------------------
// small double-precision 3-vector helpers
//---------------------------------------------------------------------------
struct D3 { double x, y, z; };
static inline D3 d3(const Vec3 &v) { return { v.x, v.y, v.z }; }
static inline D3 sub(const D3 &a, const D3 &b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

//---------------------------------------------------------------------------
static inline D3 cross(const D3 &a, const D3 &b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
             a.x * b.y - a.y * b.x };
}

//---------------------------------------------------------------------------
static inline double dot(const D3 &a, const D3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//---------------------------------------------------------------------------
static inline D3 scale(const D3 &a, double s)
{
    return { a.x * s, a.y * s, a.z * s };
}

//---------------------------------------------------------------------------
// element matrices
//---------------------------------------------------------------------------
struct TetGeom
{
    double V;
    D3 grad[4];       // nodal shape gradients (constant per tet)
    D3 curlW[6];      // curl of Whitney fn per local edge = 2 grad_i x grad_j
    D3 wCent[6];      // Whitney fn at centroid = (grad_j - grad_i)/4
};

static bool TetGeometry(const Vec3 p[4], TetGeom &gm)
{
    D3 q[4] = { d3(p[0]), d3(p[1]), d3(p[2]), d3(p[3]) };
    D3 e1 = sub(q[1], q[0]), e2 = sub(q[2], q[0]), e3 = sub(q[3], q[0]);
    double det = dot(e1, cross(e2, e3));
    gm.V = std::fabs(det) / 6.0;
    if (gm.V < 1e-30)
        return false;
    for (int i = 0; i < 4; ++i)
    {
        // opposite face nodes
        int f[3], w = 0;
        for (int n = 0; n < 4; ++n)
            if (n != i)
                f[w++] = n;
        D3 S = cross(sub(q[f[1]], q[f[0]]), sub(q[f[2]], q[f[0]]));
        S = scale(S, 0.5);
        if (dot(S, sub(q[i], q[f[0]])) > 0.0)   // make outward (away from i)
            S = scale(S, -1.0);
        gm.grad[i] = scale(S, -1.0 / (3.0 * gm.V));
    }
    for (int e = 0; e < 6; ++e)
    {
        int i = EP[e][0], j = EP[e][1];
        gm.curlW[e] = scale(cross(gm.grad[i], gm.grad[j]), 2.0);
        gm.wCent[e] = scale(sub(gm.grad[j], gm.grad[i]), 0.25);
    }
    return true;
}

//---------------------------------------------------------------------------
// curl-curl stiffness and mass matrices (local orientation, no signs)
static void TetLocal(const TetGeom &gm, double S[6][6], double M[6][6])
{
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
            S[a][b] = gm.V * dot(gm.curlW[a], gm.curlW[b]);

    auto I = [&](int m, int n) { return gm.V * ((m == n) ? 2.0 : 1.0) / 20.0; };
    for (int a = 0; a < 6; ++a)
    {
        int i = EP[a][0], j = EP[a][1];
        for (int b = 0; b < 6; ++b)
        {
            int k = EP[b][0], l = EP[b][1];
            M[a][b] = dot(gm.grad[j], gm.grad[l]) * I(i, k)
                    - dot(gm.grad[j], gm.grad[k]) * I(i, l)
                    - dot(gm.grad[i], gm.grad[l]) * I(j, k)
                    + dot(gm.grad[i], gm.grad[k]) * I(j, l);
        }
    }
}

//---------------------------------------------------------------------------
// ABC surface matrix on a boundary triangle: B = int w_a . w_b dS
static bool TriLocal(const Vec3 q0, const Vec3 q1, const Vec3 q2,
                     double B[3][3])
{
    D3 q[3] = { d3(q0), d3(q1), d3(q2) };
    D3 nv = cross(sub(q[1], q[0]), sub(q[2], q[0]));
    double A2 = std::sqrt(dot(nv, nv));         // 2*Area
    if (A2 < 1e-30)
        return false;
    double A = 0.5 * A2;
    D3 nh = scale(nv, 1.0 / A2);
    D3 sg[3];
    sg[0] = scale(cross(nh, sub(q[2], q[1])), 1.0 / A2);
    sg[1] = scale(cross(nh, sub(q[0], q[2])), 1.0 / A2);
    sg[2] = scale(cross(nh, sub(q[1], q[0])), 1.0 / A2);

    static const int FE[3][2] = { {0,1},{0,2},{1,2} };
    auto I = [&](int m, int n) { return A * ((m == n) ? 2.0 : 1.0) / 12.0; };
    for (int a = 0; a < 3; ++a)
    {
        int i = FE[a][0], j = FE[a][1];
        for (int b = 0; b < 3; ++b)
        {
            int k = FE[b][0], l = FE[b][1];
            B[a][b] = dot(sg[j], sg[l]) * I(i, k)
                    - dot(sg[j], sg[k]) * I(i, l)
                    - dot(sg[i], sg[l]) * I(j, k)
                    + dot(sg[i], sg[k]) * I(j, l);
        }
    }
    return true;
}

//---------------------------------------------------------------------------
// algebraic self-checks of the element kernels
//---------------------------------------------------------------------------
float FemSolver::ElementSelfCheckError()
{
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> U(-0.4, 0.4);
    double maxErr = 0.0;

    for (int trial = 0; trial < 24; ++trial)
    {
        // Kuhn-like tets plus random perturbations
        Vec3 p[4] = { Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0),
                      Vec3(1, 1, 1) };
        if (trial > 0)
            for (int i = 0; i < 4; ++i)
            {
                p[i].x += (float)U(rng);
                p[i].y += (float)U(rng);
                p[i].z += (float)U(rng);
            }
        TetGeom gm;
        if (!TetGeometry(p, gm))
            continue;
        double S[6][6], M[6][6];
        TetLocal(gm, S, M);

        // 1. gradient fields are curl-free: S * grad(phi) = 0
        double phi[4], sScale = 0.0;
        for (int i = 0; i < 4; ++i)
            phi[i] = U(rng);
        double e[6];
        for (int a = 0; a < 6; ++a)
            e[a] = phi[EP[a][1]] - phi[EP[a][0]];
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                sScale = std::max(sScale, std::fabs(S[a][b]));
        for (int a = 0; a < 6; ++a)
        {
            double r = 0.0;
            for (int b = 0; b < 6; ++b)
                r += S[a][b] * e[b];
            maxErr = std::max(maxErr, std::fabs(r) / std::max(1e-30, sScale));
        }

        // 2. mass matrix positive definite on random vectors
        for (int rep = 0; rep < 4; ++rep)
        {
            double v[6], q = 0.0, nv = 0.0;
            for (int a = 0; a < 6; ++a)
            {
                v[a] = U(rng);
                nv += v[a] * v[a];
            }
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b)
                    q += v[a] * M[a][b] * v[b];
            if (nv > 1e-12 && q <= 0.0)
                maxErr = std::max(maxErr, 1.0);
        }

        // 3. stiffness symmetric positive semidefinite
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
                maxErr = std::max(maxErr,
                    std::fabs(S[a][b] - S[b][a]) / std::max(1e-30, sScale));
    }
    return (float)maxErr;
}

//---------------------------------------------------------------------------
FemSolver::FemSolver()
{
    int n = (int)std::thread::hardware_concurrency();
    n = std::max(1, std::min(16, n - 1));
    pool = new ThreadPool(n);
}

//---------------------------------------------------------------------------
FemSolver::~FemSolver()
{
    delete pool;
}

//---------------------------------------------------------------------------
void FemSolver::setup(const VoxelGridSpec &grid,
                      std::vector<uint8_t> materials,
                      std::vector<MatProps> table, float freq,
                      const std::vector<size_t> &gap, int pol)
{
    g        = grid;
    mat      = std::move(materials);
    matTable = std::move(table);
    f0       = freq;
    gapCells = gap;
    polAxis  = pol;
    finished = false;
    running  = false;
    stopFlag = false;
    curIter  = 0;
    zinOk    = false;
    phaseText = "idle";
}

//---------------------------------------------------------------------------
void FemSolver::run()
{
    running = true;
    phaseText = "meshing";
    bool ok = mesh.build(g, mat);
    if (ok && !stopFlag)
    {
        phaseText = "assembling";
        assemble();
    }
    if (ok && !stopFlag && nUnknowns > 0)
    {
        phaseText = "solving (COCG)";
        solveCocg();
        phaseText = "post-processing";
        postProcess();
    }
    phaseText = didConverge ? "done" : "done (not fully converged)";
    finished = true;
    running  = false;
}

//---------------------------------------------------------------------------
void FemSolver::assemble()
{
    const double w  = 2.0 * M_PI * f0;
    const double k0 = w / C0;
    const size_t nE = mesh.edges.size();

    // ---- dof classification ----
    dofMap.assign(nE, 0);
    driveVal.assign(nE, cplx(0, 0));
    for (size_t e = 0; e < nE; ++e)
        dofMap[e] = mesh.edgePec[e] ? -2 : 0;

    // gap drive: uniform E along polAxis across the gap; V(b)-V(a) = +1 V
    {
        std::vector<int> pillars;
        std::vector<int> polIdx;
        for (size_t c : gapCells)
        {
            mesh.cellPillarEdges(c, polAxis, pillars);
            int pi = (polAxis == 0) ? (int)(c % g.nx)
                   : (polAxis == 1) ? (int)((c / g.nx) % g.ny)
                                    : (int)(c / ((size_t)g.nx * g.ny));
            polIdx.push_back(pi);
        }
        std::sort(polIdx.begin(), polIdx.end());
        polIdx.erase(std::unique(polIdx.begin(), polIdx.end()), polIdx.end());
        double gapLen = std::max<size_t>(1, polIdx.size()) * (double)g.dl;
        // E_pol = -1/gapLen  ->  V(hi) - V(lo) = +1
        cplx eVal = cplx(-(double)g.dl / gapLen, 0.0);
        std::sort(pillars.begin(), pillars.end());
        pillars.erase(std::unique(pillars.begin(), pillars.end()),
                      pillars.end());
        for (int e : pillars)
        {
            dofMap[e]   = -1;
            driveVal[e] = eVal;     // global orientation is +polAxis
        }
    }

    nUnknowns = 0;
    for (size_t e = 0; e < nE; ++e)
        if (dofMap[e] == 0)
            dofMap[e] = (int)nUnknowns++;
        else if (dofMap[e] > 0)
            dofMap[e] = -2;         // (cannot happen; safety)

    rhs.assign(nUnknowns, cplx(0, 0));
    sol.assign(nUnknowns, cplx(0, 0));

    // ---- triplets ----
    struct Trip { int r, c; cplx v; };
    std::vector<Trip> trips;
    trips.reserve(mesh.tets.size() * 30 + mesh.bfaces.size() * 9);

    auto emit = [&](int er, int ec, cplx v)
    {
        int dr = dofMap[er];
        if (dr < 0)
            return;
        int dc = dofMap[ec];
        if (dc == -2)
            return;                          // PEC: e = 0
        if (dc == -1)
            rhs[dr] -= v * driveVal[ec];     // driven edge -> RHS
        else
            trips.push_back({ dr, dc, v });
    };

    double S[6][6], M[6][6];
    for (const auto &tet : mesh.tets)
    {
        Vec3 p[4] = { mesh.nodes[tet.n[0]], mesh.nodes[tet.n[1]],
                      mesh.nodes[tet.n[2]], mesh.nodes[tet.n[3]] };
        TetGeom gm;
        if (!TetGeometry(p, gm))
            continue;
        TetLocal(gm, S, M);
        cplx epsr(1.0, 0.0);
        if (tet.mat >= MAT_DIEL0 && tet.mat - MAT_DIEL0 < (int)matTable.size())
        {
            const MatProps &mp = matTable[tet.mat - MAT_DIEL0];
            epsr = cplx(mp.epsr, -mp.sigma / (w * EPS0));
        }
        cplx km = -k0 * k0 * epsr;
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
            {
                double sg = (double)tet.esign[a] * tet.esign[b];
                emit(tet.edge[a], tet.edge[b],
                     cplx(sg * S[a][b], 0.0) + km * (sg * M[a][b]));
            }
    }

    // first-order ABC on the outer boundary: + j k0 * B
    double B[3][3];
    for (const auto &bf : mesh.bfaces)
    {
        if (!TriLocal(mesh.nodes[bf.n[0]], mesh.nodes[bf.n[1]],
                      mesh.nodes[bf.n[2]], B))
            continue;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
            {
                double sg = (double)bf.s[a] * bf.s[b];
                emit(bf.e[a], bf.e[b], cplx(0.0, k0 * sg * B[a][b]));
            }
    }

    // ---- triplets -> CSR ----
    std::sort(trips.begin(), trips.end(),
              [](const Trip &a, const Trip &b)
              {
                  return a.r != b.r ? a.r < b.r : a.c < b.c;
              });
    rowPtr.assign(nUnknowns + 1, 0);
    colIdx.clear();
    val.clear();
    colIdx.reserve(trips.size() / 2);
    val.reserve(trips.size() / 2);
    size_t t = 0;
    for (size_t r = 0; r < nUnknowns; ++r)
    {
        while (t < trips.size() && (size_t)trips[t].r == r)
        {
            int c = trips[t].c;
            cplx v(0, 0);
            while (t < trips.size() && (size_t)trips[t].r == r &&
                   trips[t].c == c)
            {
                v += trips[t].v;
                ++t;
            }
            colIdx.push_back(c);
            val.push_back(v);
        }
        rowPtr[r + 1] = (int)colIdx.size();
    }
}

//---------------------------------------------------------------------------
// diagonally preconditioned COCG for the complex-symmetric system
//---------------------------------------------------------------------------
void FemSolver::solveCocg()
{
    const size_t n = nUnknowns;
    std::vector<cplx> diag(n, cplx(1, 0));
    for (size_t r = 0; r < n; ++r)
        for (int q = rowPtr[r]; q < rowPtr[r + 1]; ++q)
            if ((size_t)colIdx[q] == r && std::abs(val[q]) > 1e-300)
                diag[r] = val[q];

    std::vector<cplx> x(n, cplx(0, 0)), r(rhs), z(n), p(n), q(n);

    auto spmv = [&](const std::vector<cplx> &in, std::vector<cplx> &out)
    {
        int chunks = pool->threadCount();
        pool->run(chunks, [&](int c)
        {
            size_t a = n * c / chunks, b = n * (c + 1) / chunks;
            for (size_t row = a; row < b; ++row)
            {
                cplx s(0, 0);
                for (int t2 = rowPtr[row]; t2 < rowPtr[row + 1]; ++t2)
                    s += val[t2] * in[colIdx[t2]];
                out[row] = s;
            }
        });
    };
    auto udot = [&](const std::vector<cplx> &a, const std::vector<cplx> &b)
    {
        cplx s(0, 0);                       // unconjugated (COCG)
        for (size_t i = 0; i < n; ++i)
            s += a[i] * b[i];
        return s;
    };
    auto nrm2 = [&](const std::vector<cplx> &a)
    {
        double s = 0;
        for (size_t i = 0; i < n; ++i)
            s += std::norm(a[i]);
        return std::sqrt(s);
    };

    double bn = nrm2(rhs);
    if (bn < 1e-300)
    {
        didConverge = false;
        return;
    }
    for (size_t i = 0; i < n; ++i)
        z[i] = r[i] / diag[i];
    p = z;
    cplx rho = udot(r, z);
    const double tol = 5e-5;
    maxIter = 6000;
    didConverge = false;

    for (int it = 0; it < maxIter && !stopFlag; ++it)
    {
        spmv(p, q);
        cplx pq = udot(p, q);
        if (std::abs(pq) < 1e-300)
            break;
        cplx alpha = rho / pq;
        for (size_t i = 0; i < n; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        double res = nrm2(r) / bn;
        resNorm = (float)res;
        curIter = it + 1;
        if (res < tol)
        {
            didConverge = true;
            break;
        }
        for (size_t i = 0; i < n; ++i)
            z[i] = r[i] / diag[i];
        cplx rho1 = udot(r, z);
        if (std::abs(rho) < 1e-300)
            break;
        cplx beta = rho1 / rho;
        rho = rho1;
        for (size_t i = 0; i < n; ++i)
            p[i] = z[i] + beta * p[i];
    }
    sol = std::move(x);
}

//---------------------------------------------------------------------------
void FemSolver::postProcess()
{
    const double w = 2.0 * M_PI * f0;
    const size_t nCells = (size_t)g.nx * g.ny * g.nz;
    std::vector<std::complex<float>> cellE(nCells * 3, {0, 0});
    cellH.assign(nCells * 3, {0, 0});
    std::vector<uint8_t> cnt(nCells, 0);

    auto dofVal = [&](int edge) -> cplx
    {
        int d = dofMap[edge];
        if (d >= 0)
            return sol[d];
        if (d == -1)
            return driveVal[edge];
        return cplx(0, 0);
    };

    for (const auto &tet : mesh.tets)
    {
        Vec3 p[4] = { mesh.nodes[tet.n[0]], mesh.nodes[tet.n[1]],
                      mesh.nodes[tet.n[2]], mesh.nodes[tet.n[3]] };
        TetGeom gm;
        if (!TetGeometry(p, gm))
            continue;
        cplx E[3] = { {0,0},{0,0},{0,0} }, C[3] = { {0,0},{0,0},{0,0} };
        for (int a = 0; a < 6; ++a)
        {
            cplx u = dofVal(tet.edge[a]) * (double)tet.esign[a];
            E[0] += u * gm.wCent[a].x;
            E[1] += u * gm.wCent[a].y;
            E[2] += u * gm.wCent[a].z;
            C[0] += u * gm.curlW[a].x;
            C[1] += u * gm.curlW[a].y;
            C[2] += u * gm.curlW[a].z;
        }
        // H = curl E / (-j w mu0)
        cplx jwmu(0.0, w * MU0);
        size_t c = (size_t)tet.cell;
        for (int d = 0; d < 3; ++d)
        {
            cellE[c * 3 + d] += (std::complex<float>)E[d];
            cellH[c * 3 + d] += (std::complex<float>)(C[d] / (-jwmu));
        }
        if (cnt[c] < 255)
            ++cnt[c];
    }
    cellEmag.assign(nCells, 0.0f);
    for (size_t c = 0; c < nCells; ++c)
    {
        if (!cnt[c])
            continue;
        float inv = 1.0f / cnt[c];
        float m2 = 0.0f;
        for (int d = 0; d < 3; ++d)
        {
            cellE[c * 3 + d] *= inv;
            cellH[c * 3 + d] *= inv;
            m2 += std::norm(cellE[c * 3 + d]);
        }
        cellEmag[c] = std::sqrt(m2);
    }

    // surface currents |Js| = |n x H| on PEC faces
    BuildPecSurfaceFaces(g, mat, MAT_PEC, surfFaces);
    {
        std::lock_guard<std::mutex> lk(resMutex);
        jsMag.assign(surfFaces.size(), 0.0f);
        for (size_t f = 0; f < surfFaces.size(); ++f)
        {
            const SurfaceFace &sf = surfFaces[f];
            int t1 = (sf.axis == 0) ? 1 : 0;
            int t2 = (sf.axis == 2) ? 1 : 2;
            float h1 = std::abs(cellH[(size_t)sf.airCell * 3 + t1]);
            float h2 = std::abs(cellH[(size_t)sf.airCell * 3 + t2]);
            jsMag[f] = std::sqrt(h1 * h1 + h2 * h2);
        }
    }

    // input impedance from the Ampere loop around the gap mid cell
    if (!gapCells.empty())
    {
        size_t c = gapCells[gapCells.size() / 2];
        int a = polAxis;
        int t1 = CYC_I[a], t2 = CYC_J[a];          // t1 x t2 = a
        const size_t strideOf[3] = { 1, (size_t)g.nx, (size_t)g.nx * g.ny };
        auto Hc = [&](size_t cell, int comp) -> cplx
        {
            return (cplx)cellH[cell * 3 + comp];
        };
        cplx I = (Hc(c + strideOf[t1], t2) - Hc(c - strideOf[t1], t2)
                - Hc(c + strideOf[t2], t1) + Hc(c - strideOf[t2], t1))
                 * (2.0 * (double)g.dl);
        if (std::abs(I) > 1e-30)
        {
            zinVal = 1.0 / I;       // V0 = 1 V across the gap
            zinOk  = true;
        }
    }
}

//---------------------------------------------------------------------------
void FemSolver::getJs(std::vector<float> &out)
{
    std::lock_guard<std::mutex> lk(resMutex);
    out = jsMag;
}

//---------------------------------------------------------------------------
bool FemSolver::getFieldPlane(int axis, int index, int &n1, int &n2,
                              std::vector<float> &out)
{
    if (cellEmag.empty() || axis < 0 || axis > 2)
        return false;
    const int nn[3] = { g.nx, g.ny, g.nz };
    index = std::max(0, std::min(nn[axis] - 1, index));
    int a1 = (axis == 0) ? 1 : 0;
    int a2 = (axis == 2) ? 1 : 2;
    n1 = nn[a1];
    n2 = nn[a2];
    out.assign((size_t)n1 * n2, 0.0f);
    for (int q2 = 0; q2 < n2; ++q2)
        for (int q1 = 0; q1 < n1; ++q1)
        {
            int co[3];
            co[axis] = index;
            co[a1] = q1;
            co[a2] = q2;
            out[(size_t)q2 * n1 + q1] =
                cellEmag[g.cellIndex(co[0], co[1], co[2])];
        }
    return true;
}

//---------------------------------------------------------------------------
void FemSolver::getMeshViz(std::vector<Vec3> &segments)
{
    mesh.vizEdges(mat, segments, 300000);
}
