//---------------------------------------------------------------------------
// GpuTlm.h - GPU (OpenGL 4.3 compute shader) execution of the TLM time loop
//---------------------------------------------------------------------------
#ifndef GpuTlmH
#define GpuTlmH

#include <string>

class TlmSolver;

// Runs the solver's configured time loop on the GPU. Returns true on
// success (all monitor/DFT/port data filled into the solver); false if the
// GPU path is unavailable or failed - 'msg' explains why and the caller
// should fall back to the CPU path.
bool RunGpuTlm(TlmSolver &s, std::string &msg);

#endif
