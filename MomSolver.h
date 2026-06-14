//---------------------------------------------------------------------------
// MomSolver.h - Method of Moments, thin-wire EFIE (phase 1)
//
// Frequency-domain surface/line integral-equation solver. Phase 1 handles
// PEC wire antennas: the scene's wire polylines (plus the feed gap) are
// welded into ordered paths, discretized into segments, and expanded in
// overlapping triangular (rooftop) basis functions. The electric-field
// integral equation with Galerkin testing and the reduced thin-wire kernel
// gives a dense complex-symmetric system Z I = V, driven by a delta-gap
// source at the feed. Solved directly (CPU) or by COCG on the GPU.
// Outputs: input impedance, wire current distribution, radiation pattern.
//---------------------------------------------------------------------------
#ifndef MomSolverH
#define MomSolverH

#include "Geometry.h"
#include "TlmSolver.h"      // ThreadPool, FarFieldData
#include <vector>
#include <complex>
#include <atomic>
#include <mutex>
#include <string>

class MomSolver
{
public:
    MomSolver();
    ~MomSolver();

    // Wire MoM. polylines are world-space PEC wires; if hasFeed, a delta-gap
    // source bridges feedA..feedB. radius<=0 picks lambda/300.
    void setupWire(const std::vector<std::vector<Vec3>> &polylines,
                   bool hasFeed, const Vec3 &feedA, const Vec3 &feedB,
                   float radius, float f0, int segPerLambda, bool useGpu);

    void run();
    void requestStop()       { stopFlag = true; }
    bool isRunning()   const { return running; }
    bool isFinished()  const { return finished; }
    int  currentStep() const { return curStep; }
    int  totalSteps()  const { return maxStep; }
    float residual()   const { return resNorm; }
    bool ranOnGpu()    const { return usedGpu; }
    const std::string &gpuStatus() const { return gpuMsg; }
    const std::string &phase() const { return phaseText; }

    std::complex<double> zin() const { return zinVal; }
    bool zinValid() const { return zinOk; }
    int  numUnknowns() const { return (int)basis.size(); }
    int  numSegments() const { return (int)segs.size(); }

    // colored wire display: 'pts' holds 2 points per segment, 'mag' one
    // |current| value per segment.
    void getWireCurrents(std::vector<Vec3> &pts, std::vector<float> &mag);
    bool computeFarField(FarFieldData &out, int nTheta = 37, int nPhi = 73);

private:
    friend bool RunGpuMom(MomSolver &s, std::string &msg);
    using cplx = std::complex<double>;

    struct Seg { int n0, n1; Vec3 t; float len; Vec3 mid; };
    // one rooftop half: a sub-segment with its ramp orientation and charge
    struct SubSeg { Vec3 a, b, t; float len; float dsign; int rampType; };
    struct Basis  { int node; SubSeg sub[2]; };

    void assemble();
    void fillMatrixCpu();
    void solveCpu();
    void postProcess();
    cplx zEntry(int m, int n) const;   // CPU EFIE entry

    // geometry / problem
    std::vector<std::vector<Vec3>> polylines;
    bool  hasFeed = false;
    Vec3  feedA, feedB;
    float radius = 0.0f;
    float f0 = 1e9f;
    int   segPerLambda = 20;
    bool  useGpu = false;

    std::vector<Vec3>  nodes;
    std::vector<Seg>   segs;
    std::vector<Basis> basis;
    int   feedBasis = -1;

    // system + solution
    std::vector<cplx>  Z;       // dense N*N (row major)
    std::vector<cplx>  V, I;
    int   N = 0;

    // results
    std::complex<double> zinVal{0, 0};
    bool   zinOk = false;
    std::vector<float> segMag;  // |current| per segment

    ThreadPool *pool = nullptr;
    std::atomic<bool> stopFlag{false};
    std::atomic<bool> running{false};
    std::atomic<bool> finished{false};
    std::atomic<int>  curStep{0};
    int   maxStep = 1;
    std::atomic<float> resNorm{0.0f};
    bool  usedGpu = false;
    std::string gpuMsg, phaseText;
};

#endif
