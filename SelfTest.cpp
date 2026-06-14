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

    fprintf(f, "%s (%d failure%s)\n", failures ? "SELFTEST FAILED"
                                               : "SELFTEST PASSED",
            failures, failures == 1 ? "" : "s");
    fclose(f);
    return failures ? 1 : 0;
}
