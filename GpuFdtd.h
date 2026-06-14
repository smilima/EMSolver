//---------------------------------------------------------------------------
// GpuFdtd.h - GPU (OpenGL 4.3 compute) execution of the FDTD time loop
//---------------------------------------------------------------------------
#ifndef GpuFdtdH
#define GpuFdtdH

#include <string>

class FdtdSolver;

// Runs the configured Yee FDTD time loop on the GPU, filling the solver's
// monitor arrays (Js DFT, Huygens DFT, ports, energy history, playback
// frames) exactly as the CPU path does. Returns true on success; on failure
// 'msg' explains why and the caller falls back to the CPU loop.
bool RunGpuFdtd(FdtdSolver &s, std::string &msg);

#endif
