//---------------------------------------------------------------------------
// FdtdSolver.h - Finite-Difference Time-Domain solver (Yee grid)
//
// Standard Yee staggered grid on the same voxel domain as the TLM solver:
//   E components on cell edges, H components on cell faces,
//   dt = 0.99 * dl / (c0 * sqrt(3))  (Courant limit)
// Materials: per-edge (epsr, sigma) averaged from the 4 adjacent cells.
// PEC: E edges touching a PEC cell are forced to zero (staircase).
// Outer boundary: first-order Mur absorbing condition on tangential E.
// Implements IFieldSolver, so it shares the whole TLM feature set in the
// UI: ports/S11, surface-current DFT, Huygens far field, energy history,
// cut plane and playback frames.
//---------------------------------------------------------------------------
#ifndef FdtdSolverH
#define FdtdSolverH

#include "TlmSolver.h"

class FdtdSolver : public IFieldSolver
{
public:
    friend bool RunGpuFdtd(FdtdSolver &s, std::string &msg);

    FdtdSolver();
    ~FdtdSolver();

    void setup(const VoxelGridSpec &grid, std::vector<uint8_t> materials,
               std::vector<MatProps> matTable, const TlmConfig &cfg);
    int  addPort(const std::vector<size_t> &cells, int polAxis, float amp);
    void addPlaneWave(int propAxis, int planeIndex, int polAxis, float amp);
    void run();
    void requestStop()        { stopFlag = true; }
    bool isRunning()    const { return running; }
    bool isFinished()   const { return finished; }
    int  currentStep()  const { return curStep; }
    int  totalSteps()   const { return config.totalSteps; }
    float currentEnergy() const { return energy; }
    bool ranOnGpu()     const { return usedGpu; }
    const std::string &gpuStatus() const { return gpuMsg; }
    const wchar_t *solverName() const { return L"FDTD"; }
    const VoxelGridSpec &grid()  const { return g; }
    const TlmConfig &cfg()       const { return config; }
    const std::vector<SurfaceFace> &faces() const { return surfFaces; }
    const std::vector<TlmPort> &ports() const { return portList; }
    float timestep() const { return dt; }
    void getJsInstant(std::vector<float> &out);
    bool getJsDft(std::vector<float> &out);
    void setFieldPlane(int axis, int index);
    bool getFieldPlane(std::vector<float> &out, int &axis, int &index,
                       int &n1, int &n2);
    bool computeFarField(FarFieldData &out, int nTheta = 37, int nPhi = 73);
    void setRecordEvery(int n) { config.recordEvery = n; }
    int  frameCount();
    bool getFrame(int idx, VizFrame &out);
    void getEnergyHistory(std::vector<int> &steps, std::vector<float> &vals);
    int  addProbe(const Vec3 &worldPos);
    int  probeCount() const { return (int)probes.size(); }
    const FieldProbe &probe(int idx) const { return probes[idx]; }

private:
    void step(int n);
    void updateH();
    void updateE();
    void applyMur();
    void monitors(int n);
    void finalizeDft();
    float waveformValue(int n) const;
    void buildCoefficients();
    void buildHuygens();

    // Yee array index helpers (sizes differ per component)
    inline size_t iEx(int i, int j, int k) const
    {
        return ((size_t)k * (g.ny + 1) + j) * g.nx + i;
    }
    inline size_t iEy(int i, int j, int k) const
    {
        return ((size_t)k * g.ny + j) * (g.nx + 1) + i;
    }
    inline size_t iEz(int i, int j, int k) const
    {
        return ((size_t)k * (g.ny + 1) + j) * (g.nx + 1) + i;
    }
    inline size_t iHx(int i, int j, int k) const
    {
        return ((size_t)k * g.ny + j) * (g.nx + 1) + i;
    }
    inline size_t iHy(int i, int j, int k) const
    {
        return ((size_t)k * (g.ny + 1) + j) * g.nx + i;
    }
    inline size_t iHz(int i, int j, int k) const
    {
        return ((size_t)k * g.ny + j) * g.nx + i;
    }

    // cell-centered field averages (for monitors)
    float ecX(int i, int j, int k) const;
    float ecY(int i, int j, int k) const;
    float ecZ(int i, int j, int k) const;
    float hcX(int i, int j, int k) const;
    float hcY(int i, int j, int k) const;
    float hcZ(int i, int j, int k) const;

    VoxelGridSpec        g;
    TlmConfig            config;
    std::vector<uint8_t> mat;
    std::vector<MatProps> matTable;

    std::vector<float>   Ex, Ey, Ez, Hx, Hy, Hz;
    // PEC masks (1 = free). ca/cb only allocated when dielectrics exist.
    std::vector<uint8_t> mEx, mEy, mEz;
    std::vector<float>   caEx, cbEx, caEy, cbEy, caEz, cbEz;
    bool                 haveDiel = false;
    float                cb0 = 0.0f;        // air update coefficient

    // Mur-1 stored boundary planes (per face, both tangential components)
    std::vector<float>   murX0a, murX0b, murX1a, murX1b;
    std::vector<float>   murY0a, murY0b, murY1a, murY1b;
    std::vector<float>   murZ0a, murZ0b, murZ1a, murZ1b;
    float                cMur = 0.0f;

    std::vector<SourceCell> sources;
    std::vector<TlmPort> portList;
    std::vector<FieldProbe> probes;
    std::vector<SurfaceFace> surfFaces;

    std::vector<float>   jsInstant, jsDftRe, jsDftIm, jsDftMag;
    int                  dftSamples = 0;

    std::vector<HuyFace> huyFaces;
    std::vector<float>   huyDft;
    int                  huySamples = 0;

    std::atomic<int>     planeAxis{-1};
    int                  planeIndex = 0;
    std::vector<float>   planeBuf;
    int                  planeN1 = 0, planeN2 = 0;

    std::vector<VizFrame> frames;
    std::vector<int>     energySteps;
    std::vector<float>   energyVals;

    std::mutex           vizMutex;
    ThreadPool          *pool = nullptr;
    std::atomic<bool>    stopFlag{false};
    std::atomic<bool>    running{false};
    std::atomic<bool>    finished{false};
    std::atomic<int>     curStep{0};
    std::atomic<float>   energy{0.0f};
    std::string          gpuMsg;
    bool                 usedGpu = false;

    float dt = 0.0f;
};

#endif
