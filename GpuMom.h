//---------------------------------------------------------------------------
// GpuMom.h - GPU (OpenGL 4.3 compute) matrix fill + COCG solve for MoM
//---------------------------------------------------------------------------
#ifndef GpuMomH
#define GpuMomH

#include <string>

class MomSolver;
class MomSurface;

// Fills the dense complex-symmetric MoM impedance matrix and solves Z I = V
// on the GPU (COCG). Writes the solution into the solver's I vector. Returns
// true on success; on failure 'msg' explains why and the caller falls back
// to the CPU fill + direct solve.
bool RunGpuMom(MomSolver &s, std::string &msg);

// Same for the surface RWG MoM (dense fill + COCG).
bool RunGpuMomSurf(MomSurface &s, std::string &msg);

#endif
