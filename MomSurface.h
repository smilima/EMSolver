//---------------------------------------------------------------------------
// MomSurface.h - Method of Moments, surface RWG EFIE (phase 2)
//
// Plane-wave scattering from triangulated PEC surfaces (plate, box, sphere,
// horn, imported STL). Interior edges of the combined triangle mesh become
// Rao-Wilton-Glisson (RWG) basis functions; the electric-field integral
// equation with Galerkin/centroid testing gives a dense complex-symmetric
// system Z J = V. Singular self-term integrals use a Duffy transform;
// non-self pairs use 7-point triangle quadrature. Solved directly (CPU) or
// by COCG on the GPU. Outputs: surface current |J|, bistatic radiation /
// RCS pattern.
//---------------------------------------------------------------------------
#ifndef MomSurfaceH
#define MomSurfaceH

#include "Geometry.h"
#include "TlmSolver.h"      // ThreadPool, FarFieldData
#include <vector>
#include <complex>
#include <atomic>
#include <mutex>
#include <string>

class MomSurface
{
public:
    MomSurface();
    ~MomSurface();

    // mesh = combined world-space PEC triangles. Plane wave travels along
    // 'propAxis' (0..2); E-field polarized along 'polAxis'.
    void setup(const TriMesh &mesh, int propAxis, int polAxis,
               float f0, bool useGpu);

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

    int numUnknowns() const { return N; }
    int numTris()     const { return (int)tris.size() / 3; }

    // per-triangle |J| for display (and the mesh it refers to)
    void getTriCurrents(std::vector<Vec3> &verts, std::vector<int> &idx,
                        std::vector<float> &triMag);
    bool computeFarField(FarFieldData &out, int nTheta = 37, int nPhi = 73);
    float matrixSymmetryError();   // |Z - Z^T| / |Z|, for self test

private:
    friend bool RunGpuMomSurf(MomSurface &s, std::string &msg);
    using cplx = std::complex<double>;

    struct Rwg { int triP, triM; Vec3 vP, vM; float len; };

    void buildRwg(const TriMesh &mesh);
    void fillCpu();
    void solveCpu();
    void excite();
    void postProcess();
    cplx zEntry(int m, int n) const;
    // ∫ over source triangle t of G and r'*G at field point fp
    void triIntegral(const Vec3 &fp, int t, bool self,
                     cplx &Ig, cplx Igv[3]) const;

    std::vector<float> verts;   // 3 per vertex
    std::vector<int>   tris;    // 3 per triangle
    std::vector<Vec3>  triC;    // centroid
    std::vector<float> triA;    // area
    std::vector<Rwg>   rwg;
    int N = 0;

    int   propAxis = 0, polAxis = 2;
    float f0 = 1e9f;
    bool  useGpu = false;

    std::vector<cplx> Z, V, J;
    std::vector<float> triMag;

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
