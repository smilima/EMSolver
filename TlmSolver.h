//---------------------------------------------------------------------------
// TlmSolver.h - 3D TLM electromagnetic solver, Symmetrical Condensed Node
//
// Theory: P.B. Johns' SCN (1987). Each cell carries 12 link-line voltage
// pulses (2 polarizations x 2 directions x 3 axes). Per time step:
//   1. inject sources into incident pulses
//   2. sample field monitors (E, H, surface currents, port V/I, Huygens DFT)
//   3. scatter (unitary 12-port scattering; dielectric cells add 3
//      open-circuit capacitive stubs, Christopoulos stub-loaded SCN)
//   4. connect (exchange pulses with neighbours; reflect at PEC/boundaries)
// dt = dl / (2 c0). PEC is modeled as voxels whose faces reflect with -1.
// Outer boundary reflects with rho (0 = matched absorbing, -1 PEC, +1 PMC).
//
// Field mapping (consistent set, used by monitors/NTFF):
//   E_j = -V_j / dl          V_j = node common voltage, polarization j
//   H_k = -ZI_k / (Z0 dl)    ZI_k = link-line circulation about axis k
//---------------------------------------------------------------------------
#ifndef TlmSolverH
#define TlmSolverH

#include "Geometry.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <functional>
#include <thread>
#include <condition_variable>
#include <cstdint>

//---------------------------------------------------------------------------
// Simple blocking thread pool: run(chunks, fn) executes fn(0..chunks-1)
// across persistent worker threads and returns when all are done.
//---------------------------------------------------------------------------
class ThreadPool
{
public:
    explicit ThreadPool(int numThreads);
    ~ThreadPool();
    void run(int chunks, const std::function<void(int)> &fn);
    int  threadCount() const { return (int)workers.size(); }

private:
    void workerMain();

    std::vector<std::thread>     workers;
    std::mutex                   m;
    std::condition_variable      cvWork, cvDone;
    const std::function<void(int)> *job = nullptr;
    std::atomic<int>             nextChunk{0};
    int                          chunksTotal = 0;
    int                          doneWorkers = 0;
    uint64_t                     generation  = 0;
    bool                         quit        = false;
};

//---------------------------------------------------------------------------
// Materials: 0 = air, 1 = PEC, >= 2 = dielectric (index matTable[id-2])
//---------------------------------------------------------------------------
enum : uint8_t { MAT_AIR = 0, MAT_PEC = 1, MAT_DIEL0 = 2 };

struct MatProps
{
    float epsr  = 1.0f;     // relative permittivity
    float sigma = 0.0f;     // conductivity S/m
};

//---------------------------------------------------------------------------
// Excitation
//---------------------------------------------------------------------------
enum class WaveformType { CwRamped, GaussianSine, GaussianPulse };

struct SourceCell
{
    size_t cell;
    int    polAxis;   // E-field axis to excite (0..2)
    float  amp;       // amplitude incl. sign
};

// A lumped port: gap cells excited along polAxis; V(t)/I(t) recorded each
// step for impedance / S11 post-processing.
struct TlmPort
{
    std::vector<size_t> cells;
    int    polAxis = 2;
    float  amp     = 1.0f;
    std::vector<float> vRec, iRec;   // per-timestep records
};

// (SurfaceFace lives in Geometry.h, shared with the FEM solver)

//---------------------------------------------------------------------------
// Huygens (NTFF) surface face: a cell on the closed recording box.
//---------------------------------------------------------------------------
struct HuyFace
{
    int    cell;
    Vec3   pos;         // position relative to domain center
    int8_t axis;        // outward normal axis
    int8_t outSign;     // outward normal sign
};

// Far-field radiation pattern over a theta/phi grid
struct FarFieldData
{
    int nTheta = 0, nPhi = 0;          // theta 0..pi, phi 0..2pi inclusive
    std::vector<float> U;              // radiation intensity, nTheta*nPhi
    float uMax        = 0.0f;
    float directivity = 0.0f;          // linear
    float peakThetaDeg = 0, peakPhiDeg = 0;
};

//---------------------------------------------------------------------------
struct TlmConfig
{
    float        f0          = 1e9f;     // excitation/analysis frequency
    WaveformType waveform    = WaveformType::CwRamped;
    int          totalSteps  = 2000;
    int          settleSteps = 600;      // DFT starts after this
    float        boundaryRho = 0.0f;     // outer boundary reflection coeff
    int          huyOffset   = 0;        // Huygens box inset (cells), 0 = off
    bool         useGpu      = false;    // try GPU compute path
    int          recordEvery = 0;        // record playback frame every N steps
};

// One recorded visualization frame for post-run playback
struct VizFrame
{
    int step = 0;
    std::vector<float> js;       // instantaneous |Js| per surface face
    std::vector<float> plane;    // |E| cut plane (may be empty)
    int planeAxis = -1, planeIdx = 0, planeN1 = 0, planeN2 = 0;
};

// E-field probe: records the 3 field components at a point every time step
// (relative units; the source amplitude sets the scale).
struct FieldProbe
{
    size_t cell = 0;
    Vec3   pos;
    std::vector<float> ex, ey, ez;
};

//---------------------------------------------------------------------------
// Common interface for the time-domain field solvers (TLM, FDTD). The UI
// drives any of them through this: same setup, monitors, playback, ports,
// Huygens far field and energy history.
//---------------------------------------------------------------------------
class IFieldSolver
{
public:
    virtual ~IFieldSolver() {}

    virtual void setup(const VoxelGridSpec &grid,
                       std::vector<uint8_t> materials,
                       std::vector<MatProps> matTable,
                       const TlmConfig &cfg) = 0;
    virtual int  addPort(const std::vector<size_t> &cells, int polAxis,
                         float amp) = 0;
    virtual void addPlaneWave(int propAxis, int planeIndex, int polAxis,
                              float amp) = 0;
    virtual void run() = 0;
    virtual void requestStop() = 0;
    virtual bool isRunning()   const = 0;
    virtual bool isFinished()  const = 0;
    virtual int  currentStep() const = 0;
    virtual int  totalSteps()  const = 0;
    virtual float currentEnergy() const = 0;
    virtual bool ranOnGpu()    const = 0;
    virtual const std::string &gpuStatus() const = 0;
    virtual const wchar_t *solverName() const = 0;
    virtual const VoxelGridSpec &grid() const = 0;
    virtual const TlmConfig &cfg() const = 0;
    virtual const std::vector<SurfaceFace> &faces() const = 0;
    virtual const std::vector<TlmPort> &ports() const = 0;
    virtual float timestep() const = 0;
    virtual void getJsInstant(std::vector<float> &out) = 0;
    virtual bool getJsDft(std::vector<float> &out) = 0;
    virtual void setFieldPlane(int axis, int index) = 0;
    virtual bool getFieldPlane(std::vector<float> &out, int &axis,
                               int &index, int &n1, int &n2) = 0;
    virtual bool computeFarField(FarFieldData &out, int nTheta = 37,
                                 int nPhi = 73) = 0;
    virtual void setRecordEvery(int n) = 0;
    virtual int  frameCount() = 0;
    virtual bool getFrame(int idx, VizFrame &out) = 0;
    virtual void getEnergyHistory(std::vector<int> &steps,
                                  std::vector<float> &vals) = 0;
    // E-field probe at a world point; returns its index (-1 if outside)
    virtual int  addProbe(const Vec3 &worldPos) = 0;
    virtual int  probeCount() const = 0;
    virtual const FieldProbe &probe(int idx) const = 0;
};

// Far-field transform from accumulated Huygens phasors (8 floats per face:
// E1re,E1im,E2re,E2im, eta0*H1re,im, eta0*H2re,im). Shared by the solvers.
bool ComputeFarFieldFromHuygens(const std::vector<HuyFace> &huyFaces,
                                const std::vector<float> &huyDft,
                                int huySamples, float f0, ThreadPool *pool,
                                FarFieldData &out, int nTheta, int nPhi);

//---------------------------------------------------------------------------
class TlmSolver : public IFieldSolver
{
public:
    TlmSolver();
    ~TlmSolver();

    void setup(const VoxelGridSpec &grid, std::vector<uint8_t> materials,
               std::vector<MatProps> matTable, const TlmConfig &cfg);

    // Lumped gap port (records V/I). Returns port index.
    int  addPort(const std::vector<size_t> &cells, int polAxis, float amp);

    // Plane wave: soft E source over the full plane axis==planeIndex
    void addPlaneWave(int propAxis, int planeIndex, int polAxis, float amp);

    // run loop control (call run() from a worker thread)
    void run();
    void requestStop()        { stopFlag = true; }
    bool isRunning()    const { return running; }
    bool isFinished()   const { return finished; }
    int  currentStep()  const { return curStep; }
    int  totalSteps()   const { return config.totalSteps; }
    float currentEnergy() const { return energy; }
    bool ranOnGpu()     const { return usedGpu; }
    const std::string &gpuStatus() const { return gpuMsg; }
    const wchar_t *solverName() const { return L"TLM"; }

    const VoxelGridSpec &grid()  const { return g; }
    const TlmConfig &cfg()       const { return config; }
    const std::vector<uint8_t> &materials() const { return mat; }
    const std::vector<SurfaceFace> &faces() const { return surfFaces; }
    const std::vector<TlmPort> &ports() const { return portList; }
    float timestep() const { return dt; }

    // Visualization snapshots (thread safe). Values are |Js| per face.
    void getJsInstant(std::vector<float> &out);
    bool getJsDft(std::vector<float> &out);

    // Field cut plane: |E| sampled over plane normal to 'axis' at 'index'.
    void setFieldPlane(int axis, int index);   // axis -1 disables
    bool getFieldPlane(std::vector<float> &out, int &axis, int &index,
                       int &n1, int &n2);

    // Far field from the Huygens surface DFT (call after the run finished).
    // Returns false if no Huygens data was recorded.
    bool computeFarField(FarFieldData &out, int nTheta = 37, int nPhi = 73);

    // Playback frames (recorded when cfg.recordEvery > 0)
    void setRecordEvery(int n) { config.recordEvery = n; }  // before run()
    int  frameCount();
    bool getFrame(int idx, VizFrame &out);

    // total link-line energy history, sampled every 16 steps
    void getEnergyHistory(std::vector<int> &steps, std::vector<float> &vals);

    // E-field probes
    int  addProbe(const Vec3 &worldPos);
    int  probeCount() const { return (int)probes.size(); }
    const FieldProbe &probe(int idx) const { return probes[idx]; }

    // Self tests
    static float ScatterUnitarityError();              // air node
    static float DielScatterUnitarityError(float epsr); // stub-loaded node

private:
    friend class GpuTlm;

    void runCpu();
    void step(int n);
    void scatter();
    void connect();
    void monitors(int n);
    void finalizeDft();
    float waveformValue(int n) const;
    void buildSurfaceFaces();
    void buildHuygens();

    static inline size_t portBase(size_t cell) { return cell * 12; }

    VoxelGridSpec        g;
    TlmConfig            config;
    std::vector<uint8_t> mat;
    std::vector<MatProps> matTable;
    std::vector<float>   yhat, ghat;   // per dielectric id: stub admittances
    std::vector<float>   V;            // 12 link pulses per cell
    std::vector<float>   Vs;           // 3 capacitive stub pulses (diel only)
    std::vector<SourceCell> sources;   // plain soft sources (plane wave)
    std::vector<TlmPort> portList;
    std::vector<FieldProbe> probes;
    std::vector<SurfaceFace> surfFaces;

    // surface current monitors
    std::vector<float>   jsInstant;
    std::vector<float>   jsDftRe, jsDftIm;     // 2 tangential comps per face
    int                  dftSamples = 0;
    std::vector<float>   jsDftMag;

    // Huygens DFT: per face 8 floats {E1re,E1im,E2re,E2im,H1re,H1im,H2re,H2im}
    std::vector<HuyFace> huyFaces;
    std::vector<float>   huyDft;
    int                  huySamples = 0;

    // field cut plane
    std::atomic<int>     planeAxis{-1};
    int                  planeIndex = 0;
    std::vector<float>   planeBuf;
    int                  planeN1 = 0, planeN2 = 0;

    // playback recording
    std::vector<VizFrame> frames;

    // energy history
    std::vector<int>   energySteps;
    std::vector<float> energyVals;

    std::mutex           vizMutex;

    ThreadPool          *pool = nullptr;
    std::atomic<bool>    stopFlag{false};
    std::atomic<bool>    running{false};
    std::atomic<bool>    finished{false};
    std::atomic<int>     curStep{0};
    std::atomic<float>   energy{0.0f};
    bool                 usedGpu = false;
    std::string          gpuMsg;

    float dt = 0.0f;
};

#endif
