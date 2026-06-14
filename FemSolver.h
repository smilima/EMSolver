//---------------------------------------------------------------------------
// FemSolver.h - frequency-domain vector FEM (Nedelec/Whitney edge elements)
//
// Solves  curl(curl E) - k0^2 (epsr - j sigma/(w eps0)) E = 0  at f0 on the
// tetrahedral mesh from FemMesh, with:
//   - tangential E = 0 on PEC edges
//   - first-order absorbing boundary (Sommerfeld) on the outer box
//   - voltage-gap drive: Dirichlet E on the feed-gap pillar edges
// The complex-symmetric system is solved by diagonally preconditioned COCG.
// Post-processing: per-cell E/H phasors, |Js| on PEC faces, |E| cut plane,
// and the input impedance Zin = V0 / I(Ampere loop) at f0.
//---------------------------------------------------------------------------
#ifndef FemSolverH
#define FemSolverH

#include "FemMesh.h"
#include "TlmSolver.h"      // MatProps, ThreadPool, SurfaceFace
#include <complex>
#include <atomic>
#include <mutex>
#include <string>

class FemSolver
{
public:
    FemSolver();
    ~FemSolver();

    // gapCells: carved feed cells (uniform E impressed along polAxis)
    void setup(const VoxelGridSpec &grid, std::vector<uint8_t> materials,
               std::vector<MatProps> matTable, float f0,
               const std::vector<size_t> &gapCells, int polAxis,
               bool useGpu = false);

    void run();                 // mesh -> assemble -> solve -> post (worker)
    void requestStop()        { stopFlag = true; }
    bool ranOnGpu()     const { return usedGpu; }
    const std::string &gpuStatus() const { return gpuMsg; }
    bool isRunning()    const { return running; }
    bool isFinished()   const { return finished; }
    int  currentStep()  const { return curIter; }
    int  totalSteps()   const { return maxIter; }
    float residual()    const { return resNorm; }
    bool converged()    const { return didConverge; }
    const std::string &phase() const { return phaseText; }

    // results (valid when finished)
    const std::vector<SurfaceFace> &faces() const { return surfFaces; }
    void getJs(std::vector<float> &out);
    bool getFieldPlane(int axis, int index, int &n1, int &n2,
                       std::vector<float> &out);
    std::complex<double> zin() const { return zinVal; }
    bool zinValid() const { return zinOk; }

    // mesh info
    size_t numNodes() const { return mesh.nodes.size(); }
    size_t numTets()  const { return mesh.tets.size(); }
    size_t numEdges() const { return mesh.edges.size(); }
    size_t numUnknowns() const { return nUnknowns; }
    void getMeshViz(std::vector<Vec3> &segments);

    // algebraic self-checks of the element assembly; returns max error:
    //  - curl-curl stiffness annihilates gradient fields
    //  - mass matrix is SPD on random vectors
    static float ElementSelfCheckError();

private:
    using cplx = std::complex<double>;
    friend bool RunGpuFemCocg(FemSolver &s, std::string &msg);

    void assemble();
    void solveCocg();
    void postProcess();

    VoxelGridSpec        g;
    std::vector<uint8_t> mat;
    std::vector<MatProps> matTable;
    float                f0 = 1e9f;
    std::vector<size_t>  gapCells;
    int                  polAxis = 2;
    bool                 useGpu = false;
    bool                 usedGpu = false;
    std::string          gpuMsg;

    FemMesh              mesh;

    // dof handling: -2 PEC, -1 driven (value in driveVal), >=0 unknown index
    std::vector<int>     dofMap;
    std::vector<cplx>    driveVal;     // per edge (driven edges only)
    size_t               nUnknowns = 0;

    // CSR (reduced system)
    std::vector<int>     rowPtr, colIdx;
    std::vector<cplx>    val;
    std::vector<cplx>    rhs, sol;

    // results
    std::vector<SurfaceFace> surfFaces;
    std::vector<float>   jsMag;
    std::vector<float>   cellEmag;             // |E| per cell
    std::vector<std::complex<float>> cellH;    // 3 per cell
    std::complex<double> zinVal{0, 0};
    bool                 zinOk = false;

    ThreadPool          *pool = nullptr;
    std::mutex           resMutex;
    std::atomic<bool>    stopFlag{false};
    std::atomic<bool>    running{false};
    std::atomic<bool>    finished{false};
    std::atomic<int>     curIter{0};
    int                  maxIter = 5000;
    std::atomic<float>   resNorm{1.0f};
    bool                 didConverge = false;
    std::string          phaseText;
};

#endif
