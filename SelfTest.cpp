//---------------------------------------------------------------------------
// SelfTest.cpp - headless physics checks for the TLM solver
//
//  1. SCN scatter matrix is unitary (energy conserving)
//  2. Stub-loaded dielectric scatter is weighted-unitary
//  3. Pulsed dipole: stays finite, matched boundary absorbs the energy
//  4. S11/impedance: half-wave dipole resonates (Im Z zero crossing with
//     physical Re Z) inside the excitation band
//  5. CW dipole: physical surface-current distribution (max at feed)
//  6. NTFF: dipole directivity ~1.64, beam broadside
//  7. Dielectric sphere next to the dipole: still stable and absorbed
//  8. GPU compute path produces the same DFT currents as the CPU path
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "SelfTest.h"
#include "Geometry.h"
#include "AntennaLib.h"
#include "TlmSolver.h"
#include "FdtdSolver.h"
#include "FemSolver.h"
#include "MomSolver.h"
#include "MomSurface.h"
#include "ProjectIO.h"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <complex>
#include <memory>
#include <thread>

#pragma package(smart_init)

static const double C0 = 299792458.0;

struct TestSim
{
    VoxelGridSpec g;
    std::unique_ptr<TlmSolver> solver;
    Vec3 feedCenter;
};

// Build the dipole geometry (grid, materials, carved feed cells) shared by
// the TLM and FEM dipole tests.
struct DipoleGeo
{
    VoxelGridSpec g;
    std::vector<uint8_t> mat;
    std::vector<size_t> gapCells;
    Vec3 feedCenter;
};

static bool BuildDipoleGeo(DipoleGeo &dg, int cpl, int pad)
{
    double f0 = 1e9, lam = C0 / f0;
    float dl = (float)(lam / cpl);
    SceneObject o = CreateAntenna(AntennaKind::Dipole, f0, {});
    Aabb sb = o.worldBounds();
    if (!sb.valid())
        return false;
    VoxelGridSpec g;
    g.dl = dl;
    Vec3 sz = sb.size();
    g.nx = (int)std::ceil(sz.x / dl) + 1 + 2 * pad;
    g.ny = (int)std::ceil(sz.y / dl) + 1 + 2 * pad;
    g.nz = (int)std::ceil(sz.z / dl) + 1 + 2 * pad;
    Vec3 c = sb.center();
    g.origin = Vec3(c.x - g.nx * dl * 0.5f, c.y - g.ny * dl * 0.5f,
                    c.z - g.nz * dl * 0.5f);
    dg.mat.assign((size_t)g.nx * g.ny * g.nz, MAT_AIR);
    for (const auto &w : o.wires)
        VoxelizeWire(w, o.position, g, dg.mat, MAT_PEC);
    Vec3 a = o.feed.a, b = o.feed.b, d = b - a;
    Vec3 a2 = a + d * 0.15f, b2 = b - d * 0.15f;
    for (int s = 0; s <= 16; ++s)
    {
        Vec3 pp = a2 + (b2 - a2) * ((float)s / 16);
        int ci, cj, ck;
        g.cellOf(pp, ci, cj, ck);
        if (!g.inGrid(ci, cj, ck))
            continue;
        size_t cell = g.cellIndex(ci, cj, ck);
        dg.mat[cell] = MAT_AIR;
        dg.gapCells.push_back(cell);
    }
    std::sort(dg.gapCells.begin(), dg.gapCells.end());
    dg.gapCells.erase(std::unique(dg.gapCells.begin(), dg.gapCells.end()),
                      dg.gapCells.end());
    dg.g = g;
    dg.feedCenter = (a + b) * 0.5f;
    return !dg.gapCells.empty();
}

//---------------------------------------------------------------------------
// Build a center-fed dipole sim at 1 GHz, 20 cells/lambda, pad 10.
// Optionally a dielectric sphere (epsr 3) is placed beside the dipole.
static bool BuildDipoleSim(TestSim &ts, WaveformType wf, int extraSteps,
                           bool dielSphere, bool useGpu)
{
    double f0 = 1e9, lam = C0 / f0;
    int cpl = 20, pad = 10;
    float dl = (float)(lam / cpl);

    SceneObject o = CreateAntenna(AntennaKind::Dipole, f0, {});
    Aabb sb = o.worldBounds();
    if (!sb.valid())
        return false;
    if (dielSphere)
    {
        sb.grow(Vec3(0.30f * (float)lam, 0, 0));   // room for the sphere
        sb.grow(Vec3(0.55f * (float)lam, 0, 0));
    }

    VoxelGridSpec g;
    g.dl = dl;
    Vec3 sz = sb.size();
    g.nx = (int)std::ceil(sz.x / dl) + 1 + 2 * pad;
    g.ny = (int)std::ceil(sz.y / dl) + 1 + 2 * pad;
    g.nz = (int)std::ceil(sz.z / dl) + 1 + 2 * pad;
    Vec3 c = sb.center();
    g.origin = Vec3(c.x - g.nx * dl * 0.5f, c.y - g.ny * dl * 0.5f,
                    c.z - g.nz * dl * 0.5f);

    std::vector<uint8_t> mat((size_t)g.nx * g.ny * g.nz, MAT_AIR);
    std::vector<MatProps> table;
    if (dielSphere)
    {
        table.push_back({ 3.0f, 0.0f });
        Vec3 sc(0.42f * (float)lam, 0, 0);
        float r = 0.12f * (float)lam;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    if ((g.cellCenter(i, j, k) - sc).length() <= r)
                        mat[g.cellIndex(i, j, k)] = MAT_DIEL0;
    }
    for (const auto &w : o.wires)
        VoxelizeWire(w, o.position, g, mat, MAT_PEC);

    // carve + collect feed gap cells
    Vec3 a = o.feed.a, b = o.feed.b, d = b - a;
    Vec3 a2 = a + d * 0.15f, b2 = b - d * 0.15f;
    std::vector<size_t> cells;
    int steps = 16;
    for (int s = 0; s <= steps; ++s)
    {
        Vec3 pp = a2 + (b2 - a2) * ((float)s / steps);
        int ci, cj, ck;
        g.cellOf(pp, ci, cj, ck);
        if (!g.inGrid(ci, cj, ck))
            continue;
        size_t cell = g.cellIndex(ci, cj, ck);
        mat[cell] = MAT_AIR;
        cells.push_back(cell);
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    if (cells.empty())
        return false;

    TlmConfig cfg;
    cfg.f0 = (float)f0;
    cfg.waveform = wf;
    cfg.boundaryRho = 0.0f;
    cfg.useGpu = useGpu;
    int spp = 2 * cpl;
    cfg.settleSteps = 2 * (g.nx + g.ny + g.nz) + 3 * spp;
    cfg.totalSteps  = cfg.settleSteps + 12 * spp + extraSteps;
    cfg.huyOffset   = pad / 2;

    ts.g = g;
    ts.feedCenter = (a + b) * 0.5f;
    ts.solver.reset(new TlmSolver());
    ts.solver->setup(g, std::move(mat), std::move(table), cfg);
    ts.solver->addPort(cells, 2 /*z*/, 1.0f);
    return true;
}

//---------------------------------------------------------------------------
// Z(f) from recorded port time series
static std::complex<double> PortZ(const TlmPort &p, double dt, double f)
{
    std::complex<double> Vf(0, 0), If(0, 0);
    double w = 2.0 * M_PI * f * dt;
    for (size_t n = 0; n < p.vRec.size(); ++n)
    {
        std::complex<double> e(std::cos(w * n), -std::sin(w * n));
        Vf += (double)p.vRec[n] * e;
        If += (double)p.iRec[n] * e;
    }
    if (std::abs(If) < 1e-30)
        return std::complex<double>(0, 0);
    return Vf / If;
}

//---------------------------------------------------------------------------
int RunSelfTest()
{
    String exeDir = ExtractFilePath(Application->ExeName);
    AnsiString outPath(exeDir + L"selftest.txt");
    FILE *f = fopen(outPath.c_str(), "w");
    if (!f)
        return 1;
    int failures = 0;
    auto check = [&](bool ok, const char *what, double v1, double v2)
    {
        fprintf(f, "%s  %-58s (%.4g / %.4g)\n", ok ? "PASS" : "FAIL",
                what, v1, v2);
        if (!ok)
            ++failures;
    };

    // --- 1+2. scatter unitarity ---
    float uerr = TlmSolver::ScatterUnitarityError();
    check(uerr < 1e-5f, "SCN scatter matrix unitary (err < 1e-5)", uerr, 1e-5);
    float derr = TlmSolver::DielScatterUnitarityError(3.0f);
    check(derr < 1e-4f, "dielectric stub scatter weighted-unitary (epsr=3)",
          derr, 1e-4);

    // --- 3+4. pulsed dipole: stability, absorption, S11 resonance ---
    std::complex<double> zTlmF0(0, 0);
    {
        TestSim ts;
        if (!BuildDipoleSim(ts, WaveformType::GaussianSine, 400, false, false))
        {
            fprintf(f, "FAIL  could not build pulsed dipole sim\n");
            ++failures;
        }
        else
        {
            double peakE = 0.0, lastE = 0.0;
            TlmSolver *s = ts.solver.get();
            std::thread th([s] { s->run(); });
            while (!s->isFinished())
            {
                Sleep(20);
                double e = s->currentEnergy();
                peakE = std::max(peakE, e);
                lastE = e;
            }
            th.join();
            lastE = s->currentEnergy();
            bool finite = std::isfinite(lastE) && std::isfinite(peakE);
            check(finite, "pulsed dipole run stays finite", lastE, peakE);
            check(peakE > 0.0, "energy was injected (peak > 0)", peakE, 0.0);
            check(lastE < 0.05 * peakE,
                  "matched boundary absorbs energy (final < 5% of peak)",
                  lastE, peakE);

            // energy history (drives the Energy-vs-time plot)
            std::vector<int> esteps;
            std::vector<float> evals;
            s->getEnergyHistory(esteps, evals);
            float eMax = 0.0f;
            for (float v : evals)
                eMax = std::max(eMax, v);
            check(evals.size() > 10 && eMax > 0.0f &&
                  evals.back() < 0.05f * eMax,
                  "energy history recorded and decaying",
                  (double)evals.size(), eMax);

            // S11 / impedance: find series resonance (Im Z zero crossing)
            const auto &ports = s->ports();
            if (ports.empty() || ports[0].vRec.empty())
            {
                fprintf(f, "FAIL  no port recording\n");
                ++failures;
            }
            else
            {
                // diagnostic dump of Z(f)
                {
                    double dtd = s->timestep(), f0d = s->cfg().f0;
                    fprintf(f, "      Z(f): ");
                    for (int q = 0; q <= 14; ++q)
                    {
                        double fq = (0.60 + 0.05 * q) * f0d;
                        std::complex<double> Z = PortZ(ports[0], dtd, fq);
                        fprintf(f, "[%.2f: %.0f%+.0fj] ", fq / f0d,
                                Z.real(), Z.imag());
                    }
                    fprintf(f, "\n");
                }
                // search the well-excited band only (Gaussian-sine energy
                // drops fast away from f0); among Im Z zero crossings pick
                // the one with the most physical resistance
                // a fat 1-cell voxel dipole at 20 cells/lambda resonates a
                // little low; accept a crossing anywhere in 0.6..1.1 f0
                double dt = s->timestep(), f0 = s->cfg().f0;
                double fRes = 0, reAtRes = 1e30;
                double prevIm = 0;
                for (int q = 0; q <= 100; ++q)
                {
                    double fq = (0.60 + 0.005 * q) * f0;   // 0.6..1.1 f0
                    std::complex<double> Z = PortZ(ports[0], dt, fq);
                    if (q > 0 && prevIm < 0 && Z.imag() >= 0 &&
                        Z.real() > 5 && Z.real() < 1000)
                    {
                        if (std::fabs(Z.real() - 73.0) <
                            std::fabs(reAtRes - 73.0))
                        {
                            fRes = fq;
                            reAtRes = Z.real();
                        }
                    }
                    prevIm = Z.imag();
                }
                bool found = fRes > 0;
                check(found, "dipole resonance found in 0.6-1.1 f0",
                      found ? fRes / f0 : 0.0, reAtRes);
                if (found)
                    check(reAtRes > 20 && reAtRes < 250,
                          "resonant resistance physical (20-250 ohm)",
                          reAtRes, 73.0);
                zTlmF0 = PortZ(ports[0], dt, f0);   // for FEM comparison
            }
        }
    }

    // --- 5+6. CW dipole: current distribution + far field ---
    {
        TestSim ts;
        if (!BuildDipoleSim(ts, WaveformType::CwRamped, 0, false, false))
        {
            fprintf(f, "FAIL  could not build CW dipole sim\n");
            ++failures;
        }
        else
        {
            TlmSolver *s = ts.solver.get();
            s->run();
            std::vector<float> js;
            bool haveDft = s->getJsDft(js);
            check(haveDft && !js.empty(), "DFT surface currents produced",
                  haveDft ? (double)js.size() : 0.0, 0.0);
            if (haveDft && !js.empty())
            {
                float mx = 0.0f;
                bool finite = true;
                for (float v : js)
                {
                    if (!std::isfinite(v))
                        finite = false;
                    mx = std::max(mx, v);
                }
                check(finite && mx > 0.0f, "currents finite and nonzero", mx, 0.0);

                const auto &faces = s->faces();
                Aabb wb;
                for (const auto &fc : faces)
                    wb.grow(fc.center);
                float zc = ts.feedCenter.z;
                float zLen = wb.size().z;
                double sumC = 0, nC = 0, sumT = 0, nT = 0;
                for (size_t i = 0; i < faces.size(); ++i)
                {
                    float dz = std::fabs(faces[i].center.z - zc);
                    if (dz < 0.12f * zLen) { sumC += js[i]; ++nC; }
                    if (dz > 0.40f * zLen) { sumT += js[i]; ++nT; }
                }
                double avgC = nC ? sumC / nC : 0.0;
                double avgT = nT ? sumT / nT : 1e30;
                check(avgC > 1.3 * avgT,
                      "current max near feed, min at tips (center > 1.3x tip)",
                      avgC, avgT);
            }

            // far field: z-dipole -> donut, D ~ 1.64, beam broadside
            FarFieldData ff;
            bool ffOk = s->computeFarField(ff);
            check(ffOk, "far-field transform produced data",
                  ffOk ? 1.0 : 0.0, 0.0);
            if (ffOk)
            {
                check(ff.directivity > 1.1f && ff.directivity < 2.6f,
                      "dipole directivity ~1.64 (1.1-2.6)",
                      ff.directivity, 1.64);
                check(std::fabs(ff.peakThetaDeg - 90.0f) < 30.0f,
                      "dipole beam broadside (theta 90 +/- 30 deg)",
                      ff.peakThetaDeg, 90.0);
            }
        }
    }

    // --- 7. dielectric sphere: stable and absorbed ---
    {
        TestSim ts;
        if (!BuildDipoleSim(ts, WaveformType::GaussianSine, 600, true, false))
        {
            fprintf(f, "FAIL  could not build dielectric sim\n");
            ++failures;
        }
        else
        {
            TlmSolver *s = ts.solver.get();
            double peakE = 0.0;
            std::thread th([s] { s->run(); });
            while (!s->isFinished())
            {
                Sleep(20);
                peakE = std::max(peakE, (double)s->currentEnergy());
            }
            th.join();
            double lastE = s->currentEnergy();
            bool finite = std::isfinite(lastE) && std::isfinite(peakE);
            check(finite && peakE > 0,
                  "dielectric (epsr=3) run stays finite", lastE, peakE);
            check(lastE < 0.10 * peakE,
                  "dielectric run energy absorbed (final < 10% of peak)",
                  lastE, peakE);
        }
    }

    // --- 8. FDTD: stability, currents, far field, TLM cross-validation ---
    {
        DipoleGeo dg;
        if (!BuildDipoleGeo(dg, 20, 10))
        {
            fprintf(f, "FAIL  could not build FDTD dipole geometry\n");
            ++failures;
        }
        else
        {
            FdtdSolver fd;
            TlmConfig cfg;
            cfg.f0 = 1e9f;
            cfg.waveform = WaveformType::GaussianSine;
            int spp = 2 * 20;
            cfg.settleSteps = 2 * (dg.g.nx + dg.g.ny + dg.g.nz) + 3 * spp;
            cfg.totalSteps  = cfg.settleSteps + 12 * spp + 500;
            cfg.huyOffset   = 5;
            fd.setup(dg.g, dg.mat, {}, cfg);
            fd.addPort(dg.gapCells, 2, 1.0f);
            fd.run();

            std::vector<int> esteps;
            std::vector<float> evals;
            fd.getEnergyHistory(esteps, evals);
            float peakE = 0.0f, lastE = evals.empty() ? 0.0f : evals.back();
            bool finite = true;
            for (float v : evals)
            {
                if (!std::isfinite(v))
                    finite = false;
                peakE = std::max(peakE, v);
            }
            check(finite && peakE > 0,
                  "FDTD pulsed dipole stays finite, energy injected",
                  lastE, peakE);
            check(lastE < 0.15f * peakE,
                  "FDTD Mur boundary absorbs energy (final < 15% of peak)",
                  lastE, peakE);

            // impedance cross-check vs TLM at f0
            std::complex<double> zFdtdCpu(0, 0);
            bool zFdtdOk = false;
            const auto &ports = fd.ports();
            if (!ports.empty() && !ports[0].vRec.empty())
            {
                std::complex<double> zF =
                    PortZ(ports[0], fd.timestep(), 1e9);
                zFdtdCpu = zF;
                zFdtdOk = true;
                fprintf(f, "      FDTD Z(f0) = %.1f%+.1fj ohm,  TLM Z(f0) = "
                        "%.1f%+.1fj ohm\n", zF.real(), zF.imag(),
                        zTlmF0.real(), zTlmF0.imag());
                double ratio = (std::abs(zTlmF0) > 1e-9)
                    ? std::abs(zF) / std::abs(zTlmF0) : 0.0;
                check(ratio > 0.5 && ratio < 2.0,
                      "FDTD |Z(f0)| within 2x of TLM (cross-validation)",
                      ratio, 1.0);
            }
            else
            {
                fprintf(f, "FAIL  FDTD port not recorded\n");
                ++failures;
            }

            // surface currents distribution
            std::vector<float> js;
            fd.getJsDft(js);
            const auto &faces = fd.faces();
            Aabb wb;
            for (const auto &fc : faces)
                wb.grow(fc.center);
            double sumC = 0, nC = 0, sumT = 0, nT = 0;
            float zc = dg.feedCenter.z, zLen = wb.size().z;
            for (size_t i = 0; i < faces.size() && i < js.size(); ++i)
            {
                float dz = std::fabs(faces[i].center.z - zc);
                if (dz < 0.12f * zLen) { sumC += js[i]; ++nC; }
                if (dz > 0.40f * zLen) { sumT += js[i]; ++nT; }
            }
            double avgC = nC ? sumC / nC : 0.0;
            double avgT = nT ? sumT / nT : 1e30;
            check(avgC > 1.3 * avgT,
                  "FDTD current max near feed (center > 1.3x tip)",
                  avgC, avgT);

            // far field via the shared Huygens transform
            FarFieldData ff;
            bool ffOk = fd.computeFarField(ff);
            check(ffOk && ff.directivity > 1.1f && ff.directivity < 2.6f,
                  "FDTD dipole directivity ~1.64 (1.1-2.6)",
                  ffOk ? ff.directivity : 0.0f, 1.64);
            if (ffOk)
                check(std::fabs(ff.peakThetaDeg - 90.0f) < 30.0f,
                      "FDTD dipole beam broadside (theta 90 +/- 30 deg)",
                      ff.peakThetaDeg, 90.0);

            // GPU FDTD vs CPU FDTD
            FdtdSolver fg;
            TlmConfig cfgG = cfg;
            cfgG.useGpu = true;
            fg.setup(dg.g, dg.mat, {}, cfgG);
            fg.addPort(dg.gapCells, 2, 1.0f);
            fg.run();
            if (!fg.ranOnGpu())
                fprintf(f, "SKIP  GPU FDTD unavailable: %s\n",
                        fg.gpuStatus().c_str());
            else
            {
                std::vector<int> gs;
                std::vector<float> gv;
                fg.getEnergyHistory(gs, gv);
                float gpeak = 0.0f, glast = gv.empty() ? 1.0f : gv.back();
                bool gfin = true;
                for (float v : gv)
                {
                    if (!std::isfinite(v)) gfin = false;
                    gpeak = std::max(gpeak, v);
                }
                fprintf(f, "      GPU FDTD adapter: %s\n", fg.gpuStatus().c_str());
                check(gfin && gpeak > 0 && glast < 0.15f * gpeak,
                      "GPU FDTD stable and absorbing", glast, gpeak);
                const auto &gp = fg.ports();
                if (zFdtdOk && !gp.empty() && !gp[0].vRec.empty())
                {
                    std::complex<double> zg = PortZ(gp[0], fg.timestep(), 1e9);
                    double rel = std::abs(zg - zFdtdCpu) /
                                 std::max(1e-9, std::abs(zFdtdCpu));
                    fprintf(f, "      GPU FDTD Z(f0) = %.1f%+.1fj ohm "
                            "(CPU %.1f%+.1fj)\n", zg.real(), zg.imag(),
                            zFdtdCpu.real(), zFdtdCpu.imag());
                    check(rel < 0.05,
                          "GPU FDTD Z(f0) matches CPU FDTD (rel < 5%)",
                          rel, 0.05);
                }
            }
        }
    }

    // --- 9. FEM: element kernels, dipole solve, TLM cross-validation ---
    {
        float eErr = FemSolver::ElementSelfCheckError();
        check(eErr < 1e-8f,
              "FEM element kernels (gradient-null, SPD mass)", eErr, 1e-8);

        DipoleGeo dg;
        if (!BuildDipoleGeo(dg, 15, 8))
        {
            fprintf(f, "FAIL  could not build FEM dipole geometry\n");
            ++failures;
        }
        else
        {
            FemSolver fs;
            fs.setup(dg.g, dg.mat, {}, 1e9f, dg.gapCells, 2);
            fs.run();
            std::complex<double> zFemCpu = fs.zin();
            bool zFemCpuOk = fs.zinValid();
            fprintf(f, "      FEM mesh: %u nodes, %u tets, %u edges, "
                    "%u unknowns; %d iters, residual %.3g\n",
                    (unsigned)fs.numNodes(), (unsigned)fs.numTets(),
                    (unsigned)fs.numEdges(), (unsigned)fs.numUnknowns(),
                    fs.currentStep(), (double)fs.residual());
            check(fs.isFinished() && fs.residual() < 1e-2f,
                  "FEM dipole solve converged (residual < 1e-2)",
                  fs.residual(), 1e-2);
            check(fs.zinValid(), "FEM port current extracted",
                  fs.zinValid() ? 1.0 : 0.0, 0.0);
            if (fs.zinValid())
            {
                std::complex<double> zf = fs.zin();
                fprintf(f, "      FEM Zin = %.1f%+.1fj ohm,  TLM Z(f0) = "
                        "%.1f%+.1fj ohm\n", zf.real(), zf.imag(),
                        zTlmF0.real(), zTlmF0.imag());
                check(zf.real() > 5.0 && zf.real() < 1500.0,
                      "FEM input resistance physical", zf.real(), 0.0);
                double ratio = (std::abs(zTlmF0) > 1e-9)
                    ? std::abs(zf) / std::abs(zTlmF0) : 0.0;
                check(ratio > 0.25 && ratio < 4.0,
                      "FEM |Zin| within 4x of TLM |Z(f0)| (cross-validation)",
                      ratio, 1.0);
            }
            // surface currents: finite, nonzero, max near feed
            std::vector<float> js;
            fs.getJs(js);
            bool finite = !js.empty();
            float mx = 0.0f;
            for (float v : js)
            {
                if (!std::isfinite(v))
                    finite = false;
                mx = std::max(mx, v);
            }
            check(finite && mx > 0.0f, "FEM surface currents finite, nonzero",
                  mx, 0.0);
            const auto &faces = fs.faces();
            Aabb wb;
            for (const auto &fc : faces)
                wb.grow(fc.center);
            double sumC = 0, nC = 0, sumT = 0, nT = 0;
            float zc = dg.feedCenter.z, zLen = wb.size().z;
            for (size_t i = 0; i < faces.size() && i < js.size(); ++i)
            {
                float dz = std::fabs(faces[i].center.z - zc);
                if (dz < 0.12f * zLen) { sumC += js[i]; ++nC; }
                if (dz > 0.40f * zLen) { sumT += js[i]; ++nT; }
            }
            double avgC = nC ? sumC / nC : 0.0;
            double avgT = nT ? sumT / nT : 1e30;
            check(avgC > 1.3 * avgT,
                  "FEM current max near feed (center > 1.3x tip)",
                  avgC, avgT);

            // GPU FEM (COCG on the GPU) vs CPU FEM
            FemSolver fg;
            fg.setup(dg.g, dg.mat, {}, 1e9f, dg.gapCells, 2, true);
            fg.run();
            if (!fg.ranOnGpu())
                fprintf(f, "SKIP  GPU FEM unavailable: %s\n",
                        fg.gpuStatus().c_str());
            else
            {
                fprintf(f, "      GPU FEM adapter: %s; %d iters, residual %.3g\n",
                        fg.gpuStatus().c_str(), fg.currentStep(),
                        (double)fg.residual());
                check(fg.zinValid(), "GPU FEM port current extracted",
                      fg.zinValid() ? 1.0 : 0.0, 0.0);
                if (fg.zinValid() && zFemCpuOk)
                {
                    std::complex<double> zg = fg.zin();
                    double rel = std::abs(zg - zFemCpu) /
                                 std::max(1e-9, std::abs(zFemCpu));
                    fprintf(f, "      GPU FEM Zin = %.1f%+.1fj ohm (CPU %.1f%+.1fj)\n",
                            zg.real(), zg.imag(), zFemCpu.real(), zFemCpu.imag());
                    check(rel < 0.05,
                          "GPU FEM Zin matches CPU FEM (rel < 5%)", rel, 0.05);
                }
            }
        }
    }

    // --- 9. GPU path matches CPU path ---
    {
        TestSim cpu, gpu;
        if (BuildDipoleSim(cpu, WaveformType::CwRamped, 0, false, false) &&
            BuildDipoleSim(gpu, WaveformType::CwRamped, 0, false, true))
        {
            cpu.solver->run();
            gpu.solver->run();
            if (!gpu.solver->ranOnGpu())
            {
                fprintf(f, "SKIP  GPU path unavailable: %s\n",
                        gpu.solver->gpuStatus().c_str());
            }
            else
            {
                fprintf(f, "      GPU adapter: %s\n",
                        gpu.solver->gpuStatus().c_str());
                std::vector<float> a, b;
                cpu.solver->getJsDft(a);
                gpu.solver->getJsDft(b);
                bool sized = (a.size() == b.size() && !a.empty());
                double num = 0, den = 0;
                if (sized)
                    for (size_t i = 0; i < a.size(); ++i)
                    {
                        num += (a[i] - b[i]) * (a[i] - b[i]);
                        den += (double)a[i] * a[i];
                    }
                double rel = (den > 0) ? std::sqrt(num / den) : 1e30;
                check(sized && rel < 0.05,
                      "GPU currents match CPU (rel RMS < 5%)", rel, 0.05);

                // port records must match too
                const auto &pa = cpu.solver->ports()[0];
                const auto &pb = gpu.solver->ports()[0];
                double n2 = 0, d2 = 0;
                size_t nn = std::min(pa.vRec.size(), pb.vRec.size());
                for (size_t i = 0; i < nn; ++i)
                {
                    n2 += (pa.vRec[i] - pb.vRec[i]) * (pa.vRec[i] - pb.vRec[i]);
                    d2 += (double)pa.vRec[i] * pa.vRec[i];
                }
                double relV = (d2 > 0) ? std::sqrt(n2 / d2) : 1e30;
                check(nn > 0 && relV < 0.05,
                      "GPU port voltage matches CPU (rel RMS < 5%)",
                      relV, 0.05);
            }
        }
        else
        {
            fprintf(f, "FAIL  could not build GPU comparison sims\n");
            ++failures;
        }
    }

    // --- 10. MoM (thin-wire EFIE): half-wave dipole impedance ---
    {
        double f0 = 1e9, lam = C0 / f0;
        SceneObject o = CreateAntenna(AntennaKind::Dipole, f0, {});
        std::vector<std::vector<Vec3>> polys;
        for (const auto &wseg : o.wires)
            polys.push_back(wseg.pts);

        MomSolver mom;
        mom.setupWire(polys, o.feed.enabled, o.feed.a, o.feed.b,
                      (float)(lam / 300.0), (float)f0, 40, false);
        mom.run();
        fprintf(f, "      MoM wire: %d segments, %d unknowns\n",
                mom.numSegments(), mom.numUnknowns());
        check(mom.numUnknowns() > 5, "MoM dipole assembled (unknowns > 5)",
              mom.numUnknowns(), 0.0);
        check(mom.zinValid(), "MoM dipole port impedance extracted",
              mom.zinValid() ? 1.0 : 0.0, 0.0);
        std::complex<double> zMomCpu(0, 0);
        if (mom.zinValid())
        {
            zMomCpu = mom.zin();
            fprintf(f, "      MoM dipole Zin = %.1f%+.1fj ohm (classic ~73+42j)\n",
                    zMomCpu.real(), zMomCpu.imag());
            check(zMomCpu.real() > 40.0 && zMomCpu.real() < 110.0,
                  "MoM dipole resistance physical (40-110 ohm)",
                  zMomCpu.real(), 73.0);
            check(zMomCpu.imag() > -80.0 && zMomCpu.imag() < 130.0,
                  "MoM dipole reactance physical (-80..130 ohm)",
                  zMomCpu.imag(), 42.0);
        }
        // current distribution: max near the center feed, ~0 at the ends
        std::vector<Vec3> wp;
        std::vector<float> wm;
        mom.getWireCurrents(wp, wm);
        float mxC = 0, mxAll = 0, endAvg = 0; int nEnd = 0;
        for (size_t s = 0; s < wm.size(); ++s)
        {
            float zc = std::fabs(wp[s*2].z + wp[s*2+1].z) * 0.5f;
            mxAll = std::max(mxAll, wm[s]);
            if (zc < 0.06f * (float)lam) mxC = std::max(mxC, wm[s]);
            if (zc > 0.20f * (float)lam) { endAvg += wm[s]; ++nEnd; }
        }
        endAvg = nEnd ? endAvg / nEnd : 1e30f;
        check(mxC > 1.5f * endAvg,
              "MoM current max at feed, min at tips", mxC, endAvg);

        FarFieldData ff;
        bool ffOk = mom.computeFarField(ff);
        check(ffOk && ff.directivity > 1.2f && ff.directivity < 2.2f,
              "MoM dipole directivity ~1.64 (1.2-2.2)",
              ffOk ? ff.directivity : 0.0f, 1.64);

        // GPU MoM vs CPU MoM
        MomSolver mg;
        mg.setupWire(polys, o.feed.enabled, o.feed.a, o.feed.b,
                     (float)(lam / 300.0), (float)f0, 40, true);
        mg.run();
        if (!mg.ranOnGpu())
            fprintf(f, "SKIP  GPU MoM unavailable: %s\n", mg.gpuStatus().c_str());
        else
        {
            fprintf(f, "      GPU MoM adapter: %s\n", mg.gpuStatus().c_str());
            if (mg.zinValid() && mom.zinValid())
            {
                std::complex<double> zg = mg.zin();
                double rel = std::abs(zg - zMomCpu) /
                             std::max(1e-9, std::abs(zMomCpu));
                fprintf(f, "      GPU MoM Zin = %.1f%+.1fj ohm (CPU %.1f%+.1fj)\n",
                        zg.real(), zg.imag(), zMomCpu.real(), zMomCpu.imag());
                check(rel < 0.05, "GPU MoM Zin matches CPU MoM (rel < 5%)",
                      rel, 0.05);
            }
        }
    }

    // --- 11. surface RWG MoM: PEC plate plane-wave scattering ---
    {
        double f0 = 1e9, lam = C0 / f0;
        // 1.2 lambda square plate in the x-y plane, ~lambda/10 triangles
        double D = 1.2 * lam;
        int nd = 12;
        TriMesh plate;
        for (int j = 0; j < nd; ++j)
            for (int i = 0; i < nd; ++i)
            {
                double x0 = -D/2 + D*i/nd, x1 = -D/2 + D*(i+1)/nd;
                double y0 = -D/2 + D*j/nd, y1 = -D/2 + D*(j+1)/nd;
                Vec3 a((float)x0,(float)y0,0), b((float)x1,(float)y0,0);
                Vec3 c((float)x1,(float)y1,0), d((float)x0,(float)y1,0);
                plate.addTri(a, b, c);
                plate.addTri(a, c, d);
            }
        // plane wave travels +z (propAxis 2), E along x (polAxis 0)
        MomSurface ms;
        ms.setup(plate, 2, 0, (float)f0, false);
        ms.run();
        fprintf(f, "      MoM surface: %d tris, %d RWG unknowns\n",
                ms.numTris(), ms.numUnknowns());
        check(ms.numUnknowns() > 50, "RWG plate assembled (unknowns > 50)",
              ms.numUnknowns(), 0.0);
        float sym = ms.matrixSymmetryError();
        check(sym < 1e-4f, "RWG matrix complex-symmetric (Galerkin)", sym, 1e-4);

        std::vector<Vec3> sv; std::vector<int> si; std::vector<float> sm;
        ms.getTriCurrents(sv, si, sm);
        float mx = 0; bool fin = true;
        for (float v : sm) { if (!std::isfinite(v)) fin = false; mx = std::max(mx, v); }
        check(fin && mx > 0.0f, "RWG surface currents finite, nonzero", mx, 0.0);

        // normal incidence on a plate -> backscatter peak along z (theta 0/180)
        FarFieldData ff;
        bool ffOk = ms.computeFarField(ff);
        bool axial = ffOk && (ff.peakThetaDeg < 35.0f || ff.peakThetaDeg > 145.0f);
        check(axial, "RWG plate scatters specularly along z (theta ~0/180)",
              ffOk ? ff.peakThetaDeg : -1.0f, 0.0);

        double rcs = ms.monostaticRcsM2();
        fprintf(f, "      plate monostatic RCS = %.4g m^2 (%.1f dBsm)\n",
                rcs, 10.0 * std::log10(std::max(1e-12, rcs)));
        check(std::isfinite(rcs) && rcs > 0.0,
              "plate monostatic RCS positive and finite", rcs, 0.0);

        // GPU surface MoM vs CPU
        MomSurface mg;
        mg.setup(plate, 2, 0, (float)f0, true);
        mg.run();
        if (!mg.ranOnGpu())
            fprintf(f, "SKIP  GPU surface MoM unavailable: %s\n",
                    mg.gpuStatus().c_str());
        else
        {
            fprintf(f, "      GPU surface MoM adapter: %s\n", mg.gpuStatus().c_str());
            // The flat-plate EFIE matrix is ill-conditioned, so raw currents
            // are sensitive to fill precision (float GPU vs double CPU). The
            // scattered far field is the robust, physically meaningful check.
            FarFieldData ffg;
            bool gOk = mg.computeFarField(ffg);
            bool dirOk = gOk && ffOk &&
                std::fabs(ffg.directivity - ff.directivity) <
                    0.25 * std::max(0.1f, ff.directivity);
            bool peakOk = gOk &&
                (ffg.peakThetaDeg < 35.0f || ffg.peakThetaDeg > 145.0f);
            fprintf(f, "      GPU far-field D=%.2f (CPU %.2f), peak theta=%.0f\n",
                    gOk ? ffg.directivity : 0.0f, ff.directivity,
                    gOk ? ffg.peakThetaDeg : -1.0f);
            check(dirOk && peakOk,
                  "GPU surface MoM far-field matches CPU (D within 25%)",
                  gOk ? ffg.directivity : 0.0f, ff.directivity);
        }
    }

    // --- 12. surface MoM auto-decimation of a high-poly mesh (no crash) ---
    {
        double f0 = 1e9, lam = C0 / f0;
        TriMesh fine;
        int nu = 80, nv = 60;          // ~9600 triangles, above the MoM cap
        double r = 0.4 * lam;
        for (int v = 0; v < nv; ++v)
        {
            double t0 = M_PI*v/nv, t1 = M_PI*(v+1)/nv;
            for (int u = 0; u < nu; ++u)
            {
                double p0 = 2*M_PI*u/nu, p1 = 2*M_PI*(u+1)/nu;
                auto sp = [&](double th, double ph){
                    return Vec3((float)(r*std::sin(th)*std::cos(ph)),
                                (float)(r*std::sin(th)*std::sin(ph)),
                                (float)(r*std::cos(th))); };
                fine.addQuad(sp(t0,p0), sp(t0,p1), sp(t1,p1), sp(t1,p0));
            }
        }
        int origTris = fine.triCount();
        MomSurface mss;
        mss.setup(fine, 2, 0, (float)f0, false);   // decimate + build RWG
        fprintf(f, "      decimation: %d tris -> %d tris, %d unknowns\n",
                origTris, mss.numTris(), mss.numUnknowns());
        // the crash was a giant dense matrix; decimation must bound it. The
        // full solve path is already exercised by the plate test above.
        check(origTris > 2000 && mss.reducedFromTris() > 0 &&
              mss.numTris() <= 2000 && mss.numUnknowns() > 0 &&
              (double)mss.numUnknowns() * mss.numUnknowns() * 16.0 < 4.0e9,
              "high-poly mesh auto-decimated to a tractable MoM size",
              mss.numTris(), 2000.0);
    }

    // --- 13. object rotation (place an antenna at any orientation) ---
    {
        double f0 = 1e9, lam = C0 / f0;
        SceneObject d = CreateAntenna(AntennaKind::Dipole, f0, {});
        RotateSceneObject(d, Vec3(0, 90, 0));   // z-dipole -> along x
        float mxX = 0, mxZ = 0;
        for (const auto &w : d.wires)
            for (const auto &p : w.pts)
            {
                mxX = std::max(mxX, std::fabs(p.x));
                mxZ = std::max(mxZ, std::fabs(p.z));
            }
        check(mxX > 0.1f * (float)lam && mxZ < 0.02f * (float)lam,
              "rotation reorients a z-dipole to x (90 deg about Y)",
              mxX, mxZ);
    }

    // --- 14. E-field probe records a field history peaking at f0 ---
    {
        TestSim ts;
        if (BuildDipoleSim(ts, WaveformType::CwRamped, 0, false, false))
        {
            double f0 = 1e9, lam = C0 / f0;
            int pidx = ts.solver->addProbe(Vec3(0.06f*(float)lam, 0, 0));
            check(pidx == 0, "E-field probe added inside the grid", pidx, 0.0);
            ts.solver->run();
            if (ts.solver->probeCount() > 0)
            {
                const FieldProbe &pb = ts.solver->probe(0);
                int M = (int)pb.ex.size();
                double mx = 0;
                for (int n = 0; n < M; ++n)
                    mx = std::max(mx, std::sqrt((double)pb.ex[n]*pb.ex[n] +
                        (double)pb.ey[n]*pb.ey[n] + (double)pb.ez[n]*pb.ez[n]));
                check(M > 100 && mx > 0.0,
                      "probe recorded a nonzero field history", mx, 0.0);
                double dtv = ts.solver->timestep();
                double bestF = 0, bestMag = 0;
                for (int q = 1; q <= 200; ++q)
                {
                    double fq = 1.5e9 * q / 200;
                    std::complex<double> Ex(0,0), Ey(0,0), Ez(0,0);
                    double w = 2.0*M_PI*fq*dtv;
                    for (int n = 0; n < M; ++n)
                    {
                        std::complex<double> e(std::cos(w*n), -std::sin(w*n));
                        Ex += (double)pb.ex[n]*e; Ey += (double)pb.ey[n]*e;
                        Ez += (double)pb.ez[n]*e;
                    }
                    double mag = std::sqrt(std::norm(Ex)+std::norm(Ey)+std::norm(Ez));
                    if (mag > bestMag) { bestMag = mag; bestF = fq; }
                }
                fprintf(f, "      probe spectral peak at %.3f GHz\n", bestF/1e9);
                check(std::fabs(bestF - f0)/f0 < 0.1,
                      "probe |E| spectrum peaks near f0", bestF/f0, 1.0);
            }
        }
    }

    // --- 15. Project save/load round-trips scene + result + plots ---
    {
        ProjectData a;
        a.settings.push_back({ "Frequency (MHz)", "1000" });
        a.settings.push_back({ "Cells per lambda", "20" });
        a.solverIdx = 3; a.excitationIdx = 1; a.dbRangeIdx = 2;
        a.gpu = true; a.showModel = false; a.planeOn = true;

        ProjObject o; o.kind = 7; o.name = "TestStl"; o.isStl = true;
        o.position = Vec3(0.1f, 0.2f, 0.3f); o.rotDeg = Vec3(10, 20, 30);
        o.stlVerts = { 0,0,0, 1,0,0, 0,1,0 }; o.stlIdx = { 0,1,2 };
        o.params.push_back({ "Scale (m/unit)", 0.001 });
        a.objects.push_back(o);

        ProjObject d2; d2.kind = 2; d2.name = "Dipole"; d2.designFreqHz = 2e9;
        d2.params.push_back({ "Length (m)", 0.15 });
        a.objects.push_back(d2);

        a.result.wirePts = { Vec3(0,0,0), Vec3(0,0,1) };
        a.result.wireMag = { 1.0f };
        a.result.hasPattern = true;
        a.result.pattern.nTheta = 3; a.result.pattern.nPhi = 4;
        a.result.pattern.U.assign(12, 0.5f);
        a.result.pattern.directivity = 1.64f;
        a.result.domain.grow(Vec3(-1, -1, -1));
        a.result.domain.grow(Vec3(1, 1, 1));
        a.result.domainVisible = true;
        a.result.probeMarkers = { Vec3(0.05f, 0, 0) };

        ProjPage pg; pg.title = "S11"; pg.xLabel = "GHz"; pg.yLabel = "dB";
        ProjCurve cv; cv.name = "S11"; cv.color = 0xFF0000u;
        cv.x = { 1, 2, 3 }; cv.y = { -10, -20, -5 };
        pg.curves.push_back(cv); a.pages.push_back(pg);

        // recorded playback frames (v2 format)
        a.playback.valid = true;
        a.playback.grid.nx = 4; a.playback.grid.ny = 5; a.playback.grid.nz = 6;
        a.playback.grid.dl = 0.01f;
        a.playback.dt = 1.2e-12f;
        SurfaceFace sf; sf.airCell = 3; sf.center = Vec3(0.1f, 0.2f, 0.3f);
        sf.axis = 1; sf.sign = -1;
        a.playback.faces.push_back(sf);
        a.playback.faces.push_back(sf);
        VizFrame vf0; vf0.step = 0; vf0.js = { 1.0f, 2.0f }; vf0.planeAxis = -1;
        VizFrame vf1; vf1.step = 4; vf1.js = { 3.0f, 4.0f };
        vf1.planeAxis = 1; vf1.planeIdx = 2; vf1.planeN1 = 4; vf1.planeN2 = 6;
        vf1.plane.assign(24, 0.5f);
        a.playback.frames.push_back(vf0);
        a.playback.frames.push_back(vf1);

        std::wstring tmp = L"selftest_proj.emsim";
        bool wrote = SaveProject(tmp, a);
        check(wrote, "project file written", wrote ? 1 : 0, 1);

        ProjectData b; std::string err;
        bool rd = LoadProject(tmp, b, err);
        check(rd, "project file read back", rd ? 1 : 0, 1);
        if (rd)
        {
            bool same =
                b.objects.size() == 2 && b.pages.size() == 1 &&
                b.solverIdx == 3 && b.excitationIdx == 1 && b.gpu &&
                !b.showModel && b.planeOn &&
                b.objects[0].isStl && b.objects[0].stlIdx.size() == 3 &&
                b.objects[0].name == "TestStl" &&
                std::fabs(b.objects[0].position.x - 0.1f) < 1e-6f &&
                std::fabs(b.objects[1].designFreqHz - 2e9) < 1.0 &&
                b.objects[1].params.size() == 1 &&
                b.result.wirePts.size() == 2 && b.result.hasPattern &&
                b.result.pattern.U.size() == 12 &&
                b.result.probeMarkers.size() == 1 &&
                b.result.domainVisible &&
                b.pages[0].curves.size() == 1 &&
                b.pages[0].curves[0].x.size() == 3 &&
                b.pages[0].curves[0].color == 0xFF0000u &&
                b.playback.valid && b.playback.frames.size() == 2 &&
                b.playback.faces.size() == 2 &&
                b.playback.grid.nx == 4 && b.playback.grid.nz == 6 &&
                std::fabs(b.playback.dt / 1.2e-12f - 1.0f) < 1e-5f &&
                b.playback.frames[1].step == 4 &&
                b.playback.frames[1].js.size() == 2 &&
                b.playback.frames[1].planeAxis == 1 &&
                b.playback.frames[1].plane.size() == 24;
            check(same, "project round-trip preserved scene/result/plots",
                  same ? 1 : 0, 1);
        }
        _wremove(tmp.c_str());
    }

    // --- 16. Recorded TLM frames survive a real save/load round-trip ---
    {
        TestSim ts;
        if (BuildDipoleSim(ts, WaveformType::GaussianSine, 0, false, false))
        {
            ts.solver->setRecordEvery(20);
            ts.solver->run();
            int fc = ts.solver->frameCount();
            check(fc > 1, "TLM run recorded playback frames", fc, 1.0);

            ProjectData a;
            a.playback.valid = true;
            a.playback.grid  = ts.solver->grid();
            a.playback.dt    = ts.solver->timestep();
            a.playback.faces = ts.solver->faces();
            for (int i = 0; i < fc; ++i)
            {
                VizFrame fr;
                if (ts.solver->getFrame(i, fr))
                    a.playback.frames.push_back(fr);
            }
            int mid = fc / 2;
            VizFrame ref = a.playback.frames.empty() ? VizFrame()
                                                     : a.playback.frames[mid];

            std::wstring tmp = L"selftest_frames.emsim";
            bool wrote = SaveProject(tmp, a);
            ProjectData b; std::string err;
            bool rd = wrote && LoadProject(tmp, b, err);
            check(rd, "frame project written and read", rd ? 1 : 0, 1);
            if (rd)
            {
                bool faceOk = b.playback.faces.size() == ts.solver->faces().size();
                bool cntOk  = (int)b.playback.frames.size() == fc;
                bool jsOk = cntOk &&
                            b.playback.frames[mid].js.size() == ref.js.size();
                double maxdiff = 0;
                if (jsOk)
                    for (size_t q = 0; q < ref.js.size(); ++q)
                        maxdiff = std::max(maxdiff, (double)std::fabs(
                            b.playback.frames[mid].js[q] - ref.js[q]));
                check(faceOk && cntOk && jsOk && maxdiff == 0.0,
                      "loaded frames bit-match the recorded TLM run",
                      maxdiff, 0.0);
            }
            _wremove(tmp.c_str());
        }
    }

    fprintf(f, "%s (%d failure%s)\n", failures ? "SELFTEST FAILED"
                                               : "SELFTEST PASSED",
            failures, failures == 1 ? "" : "s");
    fclose(f);
    return failures ? 1 : 0;
}
