//---------------------------------------------------------------------------
// GpuFem.h - GPU (OpenGL 4.3 compute) COCG solve for the FEM system
//---------------------------------------------------------------------------
#ifndef GpuFemH
#define GpuFemH

#include <string>

class FemSolver;

// Solves the assembled complex-symmetric CSR system on the GPU with a
// diagonally preconditioned COCG (single precision). Writes the solution
// into the solver's 'sol' vector. Returns true on success; on failure
// 'msg' explains why and the caller should fall back to the CPU COCG.
bool RunGpuFemCocg(FemSolver &s, std::string &msg);

#endif
