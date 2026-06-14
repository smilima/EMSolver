//---------------------------------------------------------------------------
// GpuCompute.cpp - shared OpenGL 4.3 compute context implementation
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GpuCompute.h"
#include <gl/gl.h>

#pragma package(smart_init)
#pragma comment(lib, "opengl32")

//---------------------------------------------------------------------------
// GL 4.3 entry points (loaded at runtime)
//---------------------------------------------------------------------------
typedef ptrdiff_t GLsizeiptrT;
typedef ptrdiff_t GLintptrT;
typedef char GLcharT;

#define GC_COMPUTE_SHADER             0x91B9
#define GC_SHADER_STORAGE_BUFFER      0x90D2
#define GC_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GC_COMPILE_STATUS             0x8B81
#define GC_LINK_STATUS                0x8B82
#define GC_DYNAMIC_COPY               0x88EA
#define GC_MAJOR_VERSION              0x821B
#define GC_MINOR_VERSION              0x821C

#define GC_FUNCS(X) \
    X(GLuint, glCreateShader, (GLenum)) \
    X(void, glShaderSource, (GLuint, GLsizei, const GLcharT *const *, const GLint *)) \
    X(void, glCompileShader, (GLuint)) \
    X(void, glGetShaderiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetShaderInfoLog, (GLuint, GLsizei, GLsizei *, GLcharT *)) \
    X(GLuint, glCreateProgram, (void)) \
    X(void, glAttachShader, (GLuint, GLuint)) \
    X(void, glLinkProgram, (GLuint)) \
    X(void, glGetProgramiv, (GLuint, GLenum, GLint *)) \
    X(void, glGetProgramInfoLog, (GLuint, GLsizei, GLsizei *, GLcharT *)) \
    X(void, glUseProgram, (GLuint)) \
    X(void, glDeleteShader, (GLuint)) \
    X(void, glDeleteProgram, (GLuint)) \
    X(void, glGenBuffers, (GLsizei, GLuint *)) \
    X(void, glDeleteBuffers, (GLsizei, const GLuint *)) \
    X(void, glBindBuffer, (GLenum, GLuint)) \
    X(void, glBufferData, (GLenum, GLsizeiptrT, const void *, GLenum)) \
    X(void, glBufferSubData, (GLenum, GLintptrT, GLsizeiptrT, const void *)) \
    X(void, glGetBufferSubData, (GLenum, GLintptrT, GLsizeiptrT, void *)) \
    X(void, glBindBufferBase, (GLenum, GLuint, GLuint)) \
    X(void, glDispatchCompute, (GLuint, GLuint, GLuint)) \
    X(void, glMemoryBarrier, (GLbitfield)) \
    X(GLint, glGetUniformLocation, (GLuint, const GLcharT *)) \
    X(void, glUniform1i, (GLint, GLint)) \
    X(void, glUniform1f, (GLint, GLfloat)) \
    X(void, glUniform3i, (GLint, GLint, GLint, GLint))

#define DECL_GC(ret, name, args) typedef ret (APIENTRY *name##_t) args; \
    static name##_t p##name = nullptr;
GC_FUNCS(DECL_GC)

static bool LoadGcFuncs()
{
#define LOAD_GC(ret, name, args) \
    p##name = (name##_t)wglGetProcAddress(#name); \
    if (!p##name) return false;
    GC_FUNCS(LOAD_GC)
#undef LOAD_GC
    return true;
}

//---------------------------------------------------------------------------
GpuCompute::GpuCompute()
{
    static bool registered = false;
    const wchar_t *cls = L"RFSimGpuComputeWnd";
    if (!registered)
    {
        WNDCLASSW wc = {0};
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = GetModuleHandle(0);
        wc.lpszClassName = cls;
        wc.style = CS_OWNDC;
        RegisterClassW(&wc);
        registered = true;
    }
    HWND w = CreateWindowExW(0, cls, L"", WS_POPUP, 0, 0, 4, 4, 0, 0,
                             GetModuleHandle(0), 0);
    wnd = w;
    if (!w) { errMsg = "CreateWindow failed"; return; }
    HDC d = GetDC(w);
    dc = d;
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    int pf = ChoosePixelFormat(d, &pfd);
    if (!pf || !SetPixelFormat(d, pf, &pfd))
    {
        errMsg = "SetPixelFormat failed";
        return;
    }
    HGLRC r = wglCreateContext(d);
    rc = r;
    if (!r || !wglMakeCurrent(d, r))
    {
        errMsg = "wglCreateContext failed";
        return;
    }
    GLint major = 0, minor = 0;
    glGetIntegerv(GC_MAJOR_VERSION, &major);
    glGetIntegerv(GC_MINOR_VERSION, &minor);
    if (major * 10 + minor < 43)
    {
        const char *ver = (const char *)glGetString(GL_VERSION);
        errMsg = std::string("OpenGL 4.3 required, found ") + (ver ? ver : "?");
        return;
    }
    if (!LoadGcFuncs())
    {
        errMsg = "missing GL compute entry points";
        return;
    }
    ready = true;
}

//---------------------------------------------------------------------------
GpuCompute::~GpuCompute()
{
    for (unsigned p : programs)
        if (p) pglDeleteProgram(p);
    if (!buffers.empty())
        pglDeleteBuffers((GLsizei)buffers.size(), buffers.data());
    if (rc)
    {
        wglMakeCurrent(0, 0);
        wglDeleteContext((HGLRC)rc);
    }
    if (dc && wnd)
        ReleaseDC((HWND)wnd, (HDC)dc);
    if (wnd)
        DestroyWindow((HWND)wnd);
}

//---------------------------------------------------------------------------
std::string GpuCompute::rendererName() const
{
    const char *r = (const char *)glGetString(GL_RENDERER);
    return r ? std::string(r) : std::string("unknown adapter");
}

//---------------------------------------------------------------------------
unsigned GpuCompute::buildProgram(const char *fullSource)
{
    GLuint sh = pglCreateShader(GC_COMPUTE_SHADER);
    const GLcharT *ps = fullSource;
    pglShaderSource(sh, 1, &ps, nullptr);
    pglCompileShader(sh);
    GLint ok = 0;
    pglGetShaderiv(sh, GC_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = {0};
        pglGetShaderInfoLog(sh, sizeof(log) - 1, nullptr, log);
        errMsg = std::string("shader compile: ") + log;
        pglDeleteShader(sh);
        return 0;
    }
    GLuint prog = pglCreateProgram();
    pglAttachShader(prog, sh);
    pglLinkProgram(prog);
    pglDeleteShader(sh);
    pglGetProgramiv(prog, GC_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = {0};
        pglGetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
        errMsg = std::string("shader link: ") + log;
        pglDeleteProgram(prog);
        return 0;
    }
    programs.push_back(prog);
    return prog;
}

//---------------------------------------------------------------------------
unsigned GpuCompute::makeBuffer(int binding, size_t bytes, const void *data)
{
    GLuint b = 0;
    pglGenBuffers(1, &b);
    pglBindBuffer(GC_SHADER_STORAGE_BUFFER, b);
    pglBufferData(GC_SHADER_STORAGE_BUFFER, (GLsizeiptrT)bytes, data,
                  GC_DYNAMIC_COPY);
    pglBindBufferBase(GC_SHADER_STORAGE_BUFFER, binding, b);
    buffers.push_back(b);
    return b;
}

//---------------------------------------------------------------------------
void GpuCompute::updateBuffer(unsigned buf, size_t bytes, const void *data)
{
    pglBindBuffer(GC_SHADER_STORAGE_BUFFER, buf);
    pglBufferSubData(GC_SHADER_STORAGE_BUFFER, 0, (GLsizeiptrT)bytes, data);
}

//---------------------------------------------------------------------------
void GpuCompute::readBuffer(unsigned buf, size_t bytes, void *dst)
{
    pglBindBuffer(GC_SHADER_STORAGE_BUFFER, buf);
    pglGetBufferSubData(GC_SHADER_STORAGE_BUFFER, 0, (GLsizeiptrT)bytes, dst);
}

//---------------------------------------------------------------------------
void GpuCompute::use(unsigned prog)            { pglUseProgram(prog); }
int  GpuCompute::uniform(unsigned prog, const char *name)
{
    return pglGetUniformLocation(prog, name);
}
void GpuCompute::set1i(int loc, int v)         { pglUniform1i(loc, v); }
void GpuCompute::set1f(int loc, float v)       { pglUniform1f(loc, v); }
void GpuCompute::set3i(int loc, int a, int b, int c) { pglUniform3i(loc, a, b, c); }

//---------------------------------------------------------------------------
void GpuCompute::dispatch(size_t numThreads)
{
    pglDispatchCompute((GLuint)((numThreads + 63) / 64), 1, 1);
}
void GpuCompute::dispatchGroups(unsigned groups)
{
    pglDispatchCompute(groups, 1, 1);
}
void GpuCompute::barrier()
{
    pglMemoryBarrier(GC_SHADER_STORAGE_BARRIER_BIT);
}
unsigned long GpuCompute::glErr()
{
    return glGetError();
}
