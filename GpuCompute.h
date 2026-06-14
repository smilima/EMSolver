//---------------------------------------------------------------------------
// GpuCompute.h - reusable OpenGL 4.3 compute-shader context
//
// Wraps the hidden-window GL context, runtime function loading, program/SSBO
// management, uniforms, dispatch and readback. Shared by the FDTD and FEM
// GPU solver paths (the TLM GPU path keeps its own self-contained copy).
// Construct on the worker thread, check ok(), then build programs/buffers.
//---------------------------------------------------------------------------
#ifndef GpuComputeH
#define GpuComputeH

#include <string>
#include <vector>
#include <cstddef>

class GpuCompute
{
public:
    GpuCompute();                 // creates the context; verify with ok()
    ~GpuCompute();

    bool ok() const                  { return ready; }
    const std::string &error() const { return errMsg; }
    std::string rendererName() const;

    // Compile+link a compute program from full GLSL source (caller prepends
    // its own #version/common block). Returns 0 on failure (error() set).
    unsigned buildProgram(const char *fullSource);

    // Create a std430 SSBO bound to 'binding'; data may be null (zeroed).
    unsigned makeBuffer(int binding, size_t bytes, const void *data);
    void     updateBuffer(unsigned buf, size_t bytes, const void *data);
    void     readBuffer(unsigned buf, size_t bytes, void *dst);

    void use(unsigned prog);
    int  uniform(unsigned prog, const char *name);   // -1 if absent
    void set1i(int loc, int v);
    void set1f(int loc, float v);
    void set3i(int loc, int a, int b, int c);

    void dispatch(size_t numThreads);  // groups = ceil(n / 64), local_size 64
    void dispatchGroups(unsigned groups);
    void barrier();                    // shader-storage barrier
    unsigned long glErr();             // glGetError passthrough

private:
    bool        ready = false;
    std::string errMsg;
    void       *wnd = nullptr;   // HWND
    void       *dc  = nullptr;   // HDC
    void       *rc  = nullptr;   // HGLRC
    std::vector<unsigned> programs;
    std::vector<unsigned> buffers;
};

#endif
