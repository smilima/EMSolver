//---------------------------------------------------------------------------
// GpuTlm.cpp - TLM time loop on the GPU via OpenGL 4.3 compute shaders
//
// A hidden window + GL context is created on the solver thread. The 12
// link pulses live in two ping-pong SSBOs; per step:
//   inject -> monitors (Js DFT, Huygens DFT, port V/I) -> scatter -> connect
// identical in structure and conventions to the CPU path (TlmSolver.cpp),
// so results are directly comparable (verified by the self test).
// Any failure (old GL, alloc, compile) returns false -> CPU fallback.
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GpuTlm.h"
#include "TlmSolver.h"
#include <gl/gl.h>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#pragma package(smart_init)
#pragma comment(lib, "opengl32")

//---------------------------------------------------------------------------
// GL 4.3 entry points (loaded at runtime)
//---------------------------------------------------------------------------
typedef ptrdiff_t GLsizeiptrT;
typedef ptrdiff_t GLintptrT;
typedef char GLcharT;

#define GLC_COMPUTE_SHADER            0x91B9
#define GLC_SHADER_STORAGE_BUFFER     0x90D2
#define GLC_SHADER_STORAGE_BARRIER_BIT 0x00002000
#define GLC_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GLC_COMPILE_STATUS            0x8B81
#define GLC_LINK_STATUS               0x8B82
#define GLC_INFO_LOG_LENGTH           0x8B84
#define GLC_STATIC_DRAW               0x88E4
#define GLC_DYNAMIC_COPY              0x88EA
#define GLC_MAJOR_VERSION             0x821B
#define GLC_MINOR_VERSION             0x821C

#define GL_FUNCS(X) \
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

#define DECL_GL(ret, name, args) typedef ret (APIENTRY *name##_t) args; \
    static name##_t p##name = nullptr;
GL_FUNCS(DECL_GL)

static bool LoadGlFuncs()
{
#define LOAD_GL(ret, name, args) \
    p##name = (name##_t)wglGetProcAddress(#name); \
    if (!p##name) return false;
    GL_FUNCS(LOAD_GL)
#undef LOAD_GL
    return true;
}

//---------------------------------------------------------------------------
// GLSL sources
//---------------------------------------------------------------------------
static const char *GLSL_COMMON = R"(
#version 430
layout(local_size_x = 64) in;
uniform ivec3 dims;
int cellCount() { return dims.x * dims.y * dims.z; }
int polAxisOf(int axis, int pol) {
    if (axis == 0) return pol == 0 ? 1 : 2;
    if (axis == 1) return pol == 0 ? 0 : 2;
    return pol == 0 ? 0 : 1;
}
int portOf(int axis, int side, int polAxis) {
    int first = (axis == 0) ? 1 : 0;
    int pol = (polAxis == first) ? 0 : 1;
    return axis * 4 + side * 2 + pol;
}
const ivec2 CYC[3] = ivec2[3](ivec2(1, 2), ivec2(2, 0), ivec2(0, 1));
)";

static const char *GLSL_INJECT = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 5) buffer BufSrc { ivec4 src[]; }; // cell, pol, amp(bits), -
uniform int numSrc;
uniform float sv;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(numSrc)) return;
    int cell = src[gid].x;
    int pol  = src[gid].y;
    float amp = intBitsToFloat(src[gid].z);
    float q = 0.5 * amp * sv;
    int base = cell * 12;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis == pol) continue;
        A[base + portOf(axis, 0, pol)] += q;
        A[base + portOf(axis, 1, pol)] += q;
    }
}
)";

static const char *GLSL_SCATTER = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 1) buffer BufB { float B[]; };
layout(std430, binding = 2) buffer BufMat { uint M[]; };
layout(std430, binding = 3) buffer BufProps { vec2 props[]; }; // yhat, ghat
layout(std430, binding = 4) buffer BufStub { float ST[]; };
uniform int hasStubs;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(cellCount())) return;
    int base = int(gid) * 12;
    uint mm = M[gid];
    if (mm == 1u) {            // PEC
        for (int p = 0; p < 12; ++p) B[base + p] = 0.0;
        return;
    }
    float vin[12];
    for (int p = 0; p < 12; ++p) vin[p] = A[base + p];

    float yh = 0.0, gh = 0.0;
    if (mm >= 2u) {
        vec2 pr = props[mm - 2u];
        yh = pr.x; gh = pr.y;
    }
    float Vp[3];
    float den = 4.0 + yh + gh;
    for (int j = 0; j < 3; ++j) {
        int a1 = (j == 0) ? 1 : 0;
        int a2 = (j == 2) ? 1 : 2;
        float s4 = vin[portOf(a1,0,j)] + vin[portOf(a1,1,j)] +
                   vin[portOf(a2,0,j)] + vin[portOf(a2,1,j)];
        if (mm >= 2u && hasStubs == 1) {
            float vs = ST[int(gid) * 3 + j];
            Vp[j] = 2.0 * (s4 + yh * vs) / den;
            ST[int(gid) * 3 + j] = Vp[j] - vs;
        } else
            Vp[j] = 0.5 * s4;
    }
    float ZI[3];
    for (int k = 0; k < 3; ++k) {
        int i = CYC[k].x, j = CYC[k].y;
        ZI[k] = 0.5 * (vin[portOf(i,0,j)] - vin[portOf(i,1,j)] -
                       vin[portOf(j,0,i)] + vin[portOf(j,1,i)]);
    }
    for (int i = 0; i < 3; ++i)
        for (int s = 0; s < 2; ++s)
            for (int pol = 0; pol < 2; ++pol) {
                int j = polAxisOf(i, pol);
                int k = 3 - i - j;
                float sg = ((((i + 1) % 3) == j) ? 1.0 : -1.0) *
                           ((s == 1) ? 1.0 : -1.0);
                B[base + i*4 + s*2 + pol] = Vp[j] + sg * ZI[k]
                                          - vin[portOf(i, 1 - s, j)];
            }
}
)";

static const char *GLSL_CONNECT = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 1) buffer BufB { float B[]; };
layout(std430, binding = 2) buffer BufMat { uint M[]; };
uniform float rho;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(cellCount())) return;
    int c = int(gid);
    int ci = c % dims.x;
    int cj = (c / dims.x) % dims.y;
    int ck = c / (dims.x * dims.y);
    int base = c * 12;
    bool selfPec = (M[gid] == 1u);
    ivec3 co = ivec3(ci, cj, ck);
    ivec3 strides = ivec3(1, dims.x, dims.x * dims.y);
    for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
            int q = co[a] + (s == 1 ? 1 : -1);
            for (int pol = 0; pol < 2; ++pol) {
                int p = a * 4 + s * 2 + pol;
                if (selfPec) { A[base + p] = 0.0; continue; }
                if (q < 0 || q >= dims[a]) {
                    A[base + p] = rho * B[base + p];
                } else {
                    int nb = c + (s == 1 ? strides[a] : -strides[a]);
                    if (M[nb] == 1u)
                        A[base + p] = -B[base + p];
                    else
                        A[base + p] = B[nb * 12 + a * 4 + (1 - s) * 2 + pol];
                }
            }
        }
}
)";

static const char *GLSL_MONITOR = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 6) buffer BufFaceMeta { ivec2 faceMeta[]; }; // airCell, axis
layout(std430, binding = 7) buffer BufJsInst { float jsInst[]; };
layout(std430, binding = 8) buffer BufJsDft { float jsDft[]; };
uniform int numFaces;
uniform int doDft;
uniform float cph;
uniform float sph;
float zi(int base, int k) {
    int i = CYC[k].x, j = CYC[k].y;
    return 0.5 * (A[base + portOf(i,0,j)] - A[base + portOf(i,1,j)] -
                  A[base + portOf(j,0,i)] + A[base + portOf(j,1,i)]);
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(numFaces)) return;
    int base = faceMeta[gid].x * 12;
    int ax = faceMeta[gid].y;
    int t1 = polAxisOf(ax, 0), t2 = polAxisOf(ax, 1);
    float h1 = zi(base, t1), h2 = zi(base, t2);
    jsInst[gid] = sqrt(h1 * h1 + h2 * h2);
    if (doDft == 1) {
        jsDft[gid * 4u]      += h1 * cph;
        jsDft[gid * 4u + 1u] -= h1 * sph;
        jsDft[gid * 4u + 2u] += h2 * cph;
        jsDft[gid * 4u + 3u] -= h2 * sph;
    }
}
)";

static const char *GLSL_HUYGENS = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 9)  buffer BufHuyMeta { ivec2 huyMeta[]; }; // cell, axis
layout(std430, binding = 10) buffer BufHuyDft { float huyDft[]; };
uniform int numHuy;
uniform float cph;
uniform float sph;
float zi(int base, int k) {
    int i = CYC[k].x, j = CYC[k].y;
    return 0.5 * (A[base + portOf(i,0,j)] - A[base + portOf(i,1,j)] -
                  A[base + portOf(j,0,i)] + A[base + portOf(j,1,i)]);
}
float vj(int base, int j) {
    int a1 = (j == 0) ? 1 : 0;
    int a2 = (j == 2) ? 1 : 2;
    return 0.5 * (A[base + portOf(a1,0,j)] + A[base + portOf(a1,1,j)] +
                  A[base + portOf(a2,0,j)] + A[base + portOf(a2,1,j)]);
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(numHuy)) return;
    int base = huyMeta[gid].x * 12;
    int ax = huyMeta[gid].y;
    int t1 = polAxisOf(ax, 0), t2 = polAxisOf(ax, 1);
    float e1 = -vj(base, t1), e2 = -vj(base, t2);
    float m1 = -zi(base, t1), m2 = -zi(base, t2);
    uint o = gid * 8u;
    huyDft[o]      += e1 * cph;  huyDft[o + 1u] -= e1 * sph;
    huyDft[o + 2u] += e2 * cph;  huyDft[o + 3u] -= e2 * sph;
    huyDft[o + 4u] += m1 * cph;  huyDft[o + 5u] -= m1 * sph;
    huyDft[o + 6u] += m2 * cph;  huyDft[o + 7u] -= m2 * sph;
}
)";

static const char *GLSL_PORT = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 11) buffer BufPortCells { int portCells[]; };
layout(std430, binding = 12) buffer BufPortInfo { ivec4 portInfo[]; }; // off,count,pol,mid
layout(std430, binding = 13) buffer BufPortRec { float portRec[]; };
uniform int numPorts;
uniform int stepIdx;
float zi(int base, int k) {
    int i = CYC[k].x, j = CYC[k].y;
    return 0.5 * (A[base + portOf(i,0,j)] - A[base + portOf(i,1,j)] -
                  A[base + portOf(j,0,i)] + A[base + portOf(j,1,i)]);
}
float vj(int base, int j) {
    int a1 = (j == 0) ? 1 : 0;
    int a2 = (j == 2) ? 1 : 2;
    return 0.5 * (A[base + portOf(a1,0,j)] + A[base + portOf(a1,1,j)] +
                  A[base + portOf(a2,0,j)] + A[base + portOf(a2,1,j)]);
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(numPorts)) return;
    ivec4 inf = portInfo[gid];
    float vSum = 0.0;
    for (int c = 0; c < inf.y; ++c)
        vSum += vj(portCells[inf.x + c] * 12, inf.z);
    int a = inf.z;
    int t1 = CYC[a].x, t2 = CYC[a].y;
    ivec3 strides = ivec3(1, dims.x, dims.x * dims.y);
    int mid = inf.w;
    float ziv = 0.0;
    ziv += zi((mid + strides[t1]) * 12, t2);
    ziv -= zi((mid - strides[t1]) * 12, t2);
    ziv -= zi((mid + strides[t2]) * 12, t1);
    ziv += zi((mid - strides[t2]) * 12, t1);
    float iVal = -2.0 * ziv / 376.730313;
    portRec[(uint(stepIdx) * uint(numPorts) + gid) * 2u]      = vSum;
    portRec[(uint(stepIdx) * uint(numPorts) + gid) * 2u + 1u] = iVal;
}
)";

static const char *GLSL_ENERGY = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 15) buffer BufErg { float erg[]; };
uniform int totalFloats;
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= 4096u) return;
    float s = 0.0;
    for (uint i = gid; i < uint(totalFloats); i += 4096u) {
        float v = A[i];
        s += v * v;
    }
    erg[gid] = s;
}
)";

static const char *GLSL_PLANE = R"(
layout(std430, binding = 0) buffer BufA { float A[]; };
layout(std430, binding = 2) buffer BufMat { uint M[]; };
layout(std430, binding = 14) buffer BufPlane { float plane[]; };
uniform int pAxis;
uniform int pIdx;
uniform int pN1;
uniform int pN2;
float vj(int base, int j) {
    int a1 = (j == 0) ? 1 : 0;
    int a2 = (j == 2) ? 1 : 2;
    return 0.5 * (A[base + portOf(a1,0,j)] + A[base + portOf(a1,1,j)] +
                  A[base + portOf(a2,0,j)] + A[base + portOf(a2,1,j)]);
}
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(pN1 * pN2)) return;
    int q1 = int(gid) % pN1;
    int q2 = int(gid) / pN1;
    int a1 = (pAxis == 0) ? 1 : 0;
    int a2 = (pAxis == 2) ? 1 : 2;
    ivec3 co;
    co[pAxis] = pIdx;
    co[a1] = q1;
    co[a2] = q2;
    int cell = (co.z * dims.y + co.y) * dims.x + co.x;
    if (M[cell] == 1u) { plane[gid] = 0.0; return; }
    int base = cell * 12;
    float e2 = 0.0;
    for (int j = 0; j < 3; ++j) {
        float v = vj(base, j);
        e2 += v * v;
    }
    plane[gid] = sqrt(e2);
}
)";

//---------------------------------------------------------------------------
// helpers
//---------------------------------------------------------------------------
namespace {

struct GpuCtx
{
    HWND  wnd  = 0;
    HDC   dc   = 0;
    HGLRC rc   = 0;
    std::vector<GLuint> programs, buffers;

    ~GpuCtx()
    {
        for (GLuint p : programs)
            if (p) pglDeleteProgram(p);
        if (!buffers.empty())
            pglDeleteBuffers((GLsizei)buffers.size(), buffers.data());
        if (rc)
        {
            wglMakeCurrent(0, 0);
            wglDeleteContext(rc);
        }
        if (dc)
            ReleaseDC(wnd, dc);
        if (wnd)
            DestroyWindow(wnd);
    }
};

bool CreateGlContext(GpuCtx &ctx, std::string &msg)
{
    static bool registered = false;
    const wchar_t *cls = L"RFSimGpuTlmWnd";
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
    ctx.wnd = CreateWindowExW(0, cls, L"", WS_POPUP, 0, 0, 4, 4, 0, 0,
                              GetModuleHandle(0), 0);
    if (!ctx.wnd) { msg = "CreateWindow failed"; return false; }
    ctx.dc = GetDC(ctx.wnd);
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    int pf = ChoosePixelFormat(ctx.dc, &pfd);
    if (!pf || !SetPixelFormat(ctx.dc, pf, &pfd))
    {
        msg = "SetPixelFormat failed";
        return false;
    }
    ctx.rc = wglCreateContext(ctx.dc);
    if (!ctx.rc || !wglMakeCurrent(ctx.dc, ctx.rc))
    {
        msg = "wglCreateContext failed";
        return false;
    }
    GLint major = 0, minor = 0;
    glGetIntegerv(GLC_MAJOR_VERSION, &major);
    glGetIntegerv(GLC_MINOR_VERSION, &minor);
    if (major * 10 + minor < 43)
    {
        const char *ver = (const char *)glGetString(GL_VERSION);
        msg = std::string("OpenGL 4.3 required, found ") + (ver ? ver : "?");
        return false;
    }
    if (!LoadGlFuncs())
    {
        msg = "missing GL compute entry points";
        return false;
    }
    return true;
}

//---------------------------------------------------------------------------
static std::string GlRendererName()
{
    const char *r = (const char *)glGetString(GL_RENDERER);
    return r ? std::string(r) : std::string("unknown adapter");
}

//---------------------------------------------------------------------------
GLuint BuildProgram(const char *body, std::string &msg)
{
    std::string src = std::string(GLSL_COMMON) + body;
    // the body must not re-declare #version; GLSL_COMMON has it
    GLuint sh = pglCreateShader(GLC_COMPUTE_SHADER);
    const GLcharT *ps = src.c_str();
    pglShaderSource(sh, 1, &ps, nullptr);
    pglCompileShader(sh);
    GLint ok = 0;
    pglGetShaderiv(sh, GLC_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = {0};
        pglGetShaderInfoLog(sh, sizeof(log) - 1, nullptr, log);
        msg = std::string("shader compile: ") + log;
        pglDeleteShader(sh);
        return 0;
    }
    GLuint prog = pglCreateProgram();
    pglAttachShader(prog, sh);
    pglLinkProgram(prog);
    pglDeleteShader(sh);
    pglGetProgramiv(prog, GLC_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[2048] = {0};
        pglGetProgramInfoLog(prog, sizeof(log) - 1, nullptr, log);
        msg = std::string("shader link: ") + log;
        pglDeleteProgram(prog);
        return 0;
    }
    return prog;
}

//---------------------------------------------------------------------------
GLuint MakeBuffer(GpuCtx &ctx, GLuint binding, GLsizeiptrT size,
                  const void *data)
{
    GLuint b = 0;
    pglGenBuffers(1, &b);
    pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, b);
    pglBufferData(GLC_SHADER_STORAGE_BUFFER, size, data, GLC_DYNAMIC_COPY);
    pglBindBufferBase(GLC_SHADER_STORAGE_BUFFER, binding, b);
    ctx.buffers.push_back(b);
    return b;
}

} // namespace

//---------------------------------------------------------------------------
class GpuTlm
{
public:
    static bool run(TlmSolver &s, std::string &msg);
};

bool RunGpuTlm(TlmSolver &s, std::string &msg)
{
    return GpuTlm::run(s, msg);
}

//---------------------------------------------------------------------------
bool GpuTlm::run(TlmSolver &s, std::string &msg)
{
    const size_t n = (size_t)s.g.nx * s.g.ny * s.g.nz;
    if (n == 0)
    {
        msg = "empty grid";
        return false;
    }
    if (n > 60000000)   // VRAM-bound; buffer alloc below falls back if it won't fit
    {
        msg = "grid too large for the GPU path (>60 M cells)";
        return false;
    }

    GpuCtx ctx;
    if (!CreateGlContext(ctx, msg))
        return false;

    // ---- programs ----
    GLuint prInject = BuildProgram(GLSL_INJECT, msg);   if (!prInject) return false;
    GLuint prScatter = BuildProgram(GLSL_SCATTER, msg); if (!prScatter) return false;
    GLuint prConnect = BuildProgram(GLSL_CONNECT, msg); if (!prConnect) return false;
    GLuint prMonitor = BuildProgram(GLSL_MONITOR, msg); if (!prMonitor) return false;
    GLuint prHuy = BuildProgram(GLSL_HUYGENS, msg);     if (!prHuy) return false;
    GLuint prPort = BuildProgram(GLSL_PORT, msg);       if (!prPort) return false;
    GLuint prPlane = BuildProgram(GLSL_PLANE, msg);     if (!prPlane) return false;
    GLuint prEnergy = BuildProgram(GLSL_ENERGY, msg);   if (!prEnergy) return false;
    ctx.programs = { prInject, prScatter, prConnect, prMonitor, prHuy,
                     prPort, prPlane, prEnergy };

    // ---- buffers ----
    glGetError();
    MakeBuffer(ctx, 0, (GLsizeiptrT)(n * 12 * sizeof(float)), s.V.data()); // A
    MakeBuffer(ctx, 1, (GLsizeiptrT)(n * 12 * sizeof(float)), nullptr);    // B
    std::vector<uint32_t> mat32(n);
    for (size_t i = 0; i < n; ++i)
        mat32[i] = s.mat[i];
    MakeBuffer(ctx, 2, (GLsizeiptrT)(n * 4), mat32.data());
    mat32.clear();
    mat32.shrink_to_fit();

    std::vector<float> props;
    for (size_t i = 0; i < s.yhat.size(); ++i)
    {
        props.push_back(s.yhat[i]);
        props.push_back(s.ghat[i]);
    }
    if (props.empty())
        props = { 0.0f, 0.0f };
    MakeBuffer(ctx, 3, (GLsizeiptrT)(props.size() * 4), props.data());

    bool hasStubs = !s.Vs.empty();
    MakeBuffer(ctx, 4, (GLsizeiptrT)(hasStubs ? s.Vs.size() * 4 : 4),
               hasStubs ? s.Vs.data() : nullptr);

    // sources: plane-wave cells + all port cells
    struct SrcRec { int32_t cell, pol; float amp; float pad; };
    std::vector<SrcRec> srcs;
    for (const auto &sc : s.sources)
        srcs.push_back({ (int32_t)sc.cell, sc.polAxis, sc.amp, 0 });
    for (const auto &p : s.portList)
        for (size_t c : p.cells)
            srcs.push_back({ (int32_t)c, p.polAxis, p.amp, 0 });
    int numSrc = (int)srcs.size();
    MakeBuffer(ctx, 5, (GLsizeiptrT)std::max<size_t>(1, srcs.size()) * 16,
               srcs.empty() ? nullptr : srcs.data());

    // surface faces
    const int nf = (int)s.surfFaces.size();
    {
        std::vector<int32_t> meta(2 * std::max(1, nf));
        for (int i = 0; i < nf; ++i)
        {
            meta[i * 2]     = s.surfFaces[i].airCell;
            meta[i * 2 + 1] = s.surfFaces[i].axis;
        }
        MakeBuffer(ctx, 6, (GLsizeiptrT)meta.size() * 4, meta.data());
    }
    GLuint bufJsInst = MakeBuffer(ctx, 7,
        (GLsizeiptrT)std::max(1, nf) * 4, nullptr);
    std::vector<float> zero4(std::max(1, nf) * 4, 0.0f);
    GLuint bufJsDft = MakeBuffer(ctx, 8,
        (GLsizeiptrT)zero4.size() * 4, zero4.data());

    // Huygens
    const int nh = (int)s.huyFaces.size();
    {
        std::vector<int32_t> meta(2 * std::max(1, nh));
        for (int i = 0; i < nh; ++i)
        {
            meta[i * 2]     = s.huyFaces[i].cell;
            meta[i * 2 + 1] = s.huyFaces[i].axis;
        }
        MakeBuffer(ctx, 9, (GLsizeiptrT)meta.size() * 4, meta.data());
    }
    std::vector<float> zero8(std::max(1, nh) * 8, 0.0f);
    GLuint bufHuyDft = MakeBuffer(ctx, 10,
        (GLsizeiptrT)zero8.size() * 4, zero8.data());

    // ports
    const int numPorts = (int)s.portList.size();
    {
        std::vector<int32_t> cellsFlat, info;
        for (const auto &p : s.portList)
        {
            int off = (int)cellsFlat.size();
            for (size_t c : p.cells)
                cellsFlat.push_back((int32_t)c);
            int mid = p.cells.empty() ? 0
                      : (int)p.cells[p.cells.size() / 2];
            info.push_back(off);
            info.push_back((int)p.cells.size());
            info.push_back(p.polAxis);
            info.push_back(mid);
        }
        if (cellsFlat.empty()) cellsFlat.push_back(0);
        if (info.empty()) info.assign(4, 0);
        MakeBuffer(ctx, 11, (GLsizeiptrT)cellsFlat.size() * 4, cellsFlat.data());
        MakeBuffer(ctx, 12, (GLsizeiptrT)info.size() * 4, info.data());
    }
    size_t portRecCount = (size_t)std::max(1, numPorts) * s.config.totalSteps * 2;
    GLuint bufPortRec = MakeBuffer(ctx, 13,
        (GLsizeiptrT)portRecCount * 4, nullptr);

    // field plane scratch
    int planeMax = std::max(s.g.nx * s.g.ny,
                   std::max(s.g.ny * s.g.nz, s.g.nx * s.g.nz));
    GLuint bufPlane = MakeBuffer(ctx, 14, (GLsizeiptrT)planeMax * 4, nullptr);

    // energy reduction partials
    GLuint bufErg = MakeBuffer(ctx, 15, 4096 * 4, nullptr);

    if (glGetError() != GL_NO_ERROR)
    {
        msg = "GPU buffer allocation failed (out of memory?)";
        return false;
    }

    // ---- uniforms ----
    auto setDims = [&](GLuint pr)
    {
        pglUseProgram(pr);
        pglUniform3i(pglGetUniformLocation(pr, "dims"),
                     s.g.nx, s.g.ny, s.g.nz);
    };
    for (GLuint pr : ctx.programs)
        setDims(pr);

    GLint locSv      = (pglUseProgram(prInject),
                        pglGetUniformLocation(prInject, "sv"));
    pglUniform1i(pglGetUniformLocation(prInject, "numSrc"), numSrc);
    pglUseProgram(prScatter);
    pglUniform1i(pglGetUniformLocation(prScatter, "hasStubs"), hasStubs ? 1 : 0);
    pglUseProgram(prConnect);
    pglUniform1f(pglGetUniformLocation(prConnect, "rho"), s.config.boundaryRho);
    pglUseProgram(prMonitor);
    pglUniform1i(pglGetUniformLocation(prMonitor, "numFaces"), nf);
    GLint locMonDft = pglGetUniformLocation(prMonitor, "doDft");
    GLint locMonC   = pglGetUniformLocation(prMonitor, "cph");
    GLint locMonS   = pglGetUniformLocation(prMonitor, "sph");
    pglUseProgram(prHuy);
    pglUniform1i(pglGetUniformLocation(prHuy, "numHuy"), nh);
    GLint locHuyC = pglGetUniformLocation(prHuy, "cph");
    GLint locHuyS = pglGetUniformLocation(prHuy, "sph");
    pglUseProgram(prPort);
    pglUniform1i(pglGetUniformLocation(prPort, "numPorts"), numPorts);
    GLint locPortStep = pglGetUniformLocation(prPort, "stepIdx");
    pglUseProgram(prEnergy);
    pglUniform1i(pglGetUniformLocation(prEnergy, "totalFloats"),
                 (GLint)(n * 12));

    auto groups = [](size_t count) -> GLuint
    {
        return (GLuint)((count + 63) / 64);
    };

    // ---- time loop ----
    const int total = s.config.totalSteps;
    int dftSamples = 0, huySamples = 0;
    std::vector<float> jsTmp(std::max(1, nf));
    std::vector<float> planeTmp;

    for (int nstep = 0; nstep < total && !s.stopFlag; ++nstep)
    {
        float sv = s.waveformValue(nstep);
        bool doDft = nstep >= s.config.settleSteps;
        float phase = 2.0f * (float)M_PI * s.config.f0 * (nstep * s.dt);
        float cph = std::cos(phase), sph = std::sin(phase);

        if (numSrc > 0)
        {
            pglUseProgram(prInject);
            pglUniform1f(locSv, sv);
            pglDispatchCompute(groups(numSrc), 1, 1);
            pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);
        }
        if (nf > 0)
        {
            pglUseProgram(prMonitor);
            pglUniform1i(locMonDft, doDft ? 1 : 0);
            pglUniform1f(locMonC, cph);
            pglUniform1f(locMonS, sph);
            pglDispatchCompute(groups(nf), 1, 1);
        }
        if (doDft && nh > 0)
        {
            pglUseProgram(prHuy);
            pglUniform1f(locHuyC, cph);
            pglUniform1f(locHuyS, sph);
            pglDispatchCompute(groups(nh), 1, 1);
            ++huySamples;
        }
        if (numPorts > 0)
        {
            pglUseProgram(prPort);
            pglUniform1i(locPortStep, nstep);
            pglDispatchCompute(groups(numPorts), 1, 1);
        }
        if (doDft)
            ++dftSamples;

        // total link-line energy, every 16 steps (matches the CPU path)
        if ((nstep & 15) == 0)
        {
            pglUseProgram(prEnergy);
            pglDispatchCompute(64, 1, 1);
            pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);
            float part[4096];
            pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufErg);
            pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                                sizeof(part), part);
            double e = 0.0;
            for (float p : part)
                e += p;
            s.energy = (float)e;
            std::lock_guard<std::mutex> lk(s.vizMutex);
            s.energySteps.push_back(nstep);
            s.energyVals.push_back((float)e);
        }
        pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);

        pglUseProgram(prScatter);
        pglDispatchCompute(groups(n), 1, 1);
        pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);
        pglUseProgram(prConnect);
        pglDispatchCompute(groups(n), 1, 1);
        pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);

        s.curStep = nstep + 1;

        // periodic visualization readback
        const int recEvery = s.config.recordEvery;
        bool recordNow = (recEvery > 0 && (nstep % recEvery) == 0);
        if ((nstep & 7) == 0 || nstep == total - 1 || recordNow)
        {
            std::lock_guard<std::mutex> lk(s.vizMutex);
            if (nf > 0)
            {
                pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufJsInst);
                pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                                    (GLsizeiptrT)nf * 4, s.jsInstant.data());
            }
            int pa = s.planeAxis.load();
            if (pa >= 0)
            {
                const int nn[3] = { s.g.nx, s.g.ny, s.g.nz };
                int a1 = (pa == 0) ? 1 : 0;
                int a2 = (pa == 2) ? 1 : 2;
                int n1 = nn[a1], n2 = nn[a2];
                int idx = std::min(std::max(s.planeIndex, 0), nn[pa] - 1);
                pglUseProgram(prPlane);
                pglUniform1i(pglGetUniformLocation(prPlane, "pAxis"), pa);
                pglUniform1i(pglGetUniformLocation(prPlane, "pIdx"), idx);
                pglUniform1i(pglGetUniformLocation(prPlane, "pN1"), n1);
                pglUniform1i(pglGetUniformLocation(prPlane, "pN2"), n2);
                pglDispatchCompute(groups((size_t)n1 * n2), 1, 1);
                pglMemoryBarrier(GLC_SHADER_STORAGE_BARRIER_BIT);
                s.planeN1 = n1;
                s.planeN2 = n2;
                s.planeBuf.assign((size_t)n1 * n2, 0.0f);
                pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufPlane);
                pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                                    (GLsizeiptrT)((size_t)n1 * n2 * 4),
                                    s.planeBuf.data());
            }
            if (recordNow)
            {
                VizFrame fr;
                fr.step = nstep;
                fr.js = s.jsInstant;
                if (pa >= 0 && !s.planeBuf.empty())
                {
                    fr.plane     = s.planeBuf;
                    fr.planeAxis = pa;
                    fr.planeIdx  = s.planeIndex;
                    fr.planeN1   = s.planeN1;
                    fr.planeN2   = s.planeN2;
                }
                s.frames.push_back(std::move(fr));
            }
        }
    }

    // ---- final readback into solver arrays ----
    {
        std::lock_guard<std::mutex> lk(s.vizMutex);
        if (nf > 0)
        {
            std::vector<float> dft4((size_t)nf * 4);
            pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufJsDft);
            pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                                (GLsizeiptrT)dft4.size() * 4, dft4.data());
            for (int i = 0; i < nf; ++i)
            {
                s.jsDftRe[i * 2]     = dft4[i * 4];
                s.jsDftIm[i * 2]     = dft4[i * 4 + 1];
                s.jsDftRe[i * 2 + 1] = dft4[i * 4 + 2];
                s.jsDftIm[i * 2 + 1] = dft4[i * 4 + 3];
            }
        }
        if (nh > 0)
        {
            pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufHuyDft);
            pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                                (GLsizeiptrT)s.huyDft.size() * 4,
                                s.huyDft.data());
        }
        s.dftSamples = dftSamples;
        s.huySamples = huySamples;
    }
    if (numPorts > 0)
    {
        int doneSteps = s.curStep;
        std::vector<float> rec((size_t)numPorts * doneSteps * 2);
        pglBindBuffer(GLC_SHADER_STORAGE_BUFFER, bufPortRec);
        pglGetBufferSubData(GLC_SHADER_STORAGE_BUFFER, 0,
                            (GLsizeiptrT)rec.size() * 4, rec.data());
        for (int p = 0; p < numPorts; ++p)
        {
            auto &port = s.portList[p];
            port.vRec.resize(doneSteps);
            port.iRec.resize(doneSteps);
            for (int t = 0; t < doneSteps; ++t)
            {
                port.vRec[t] = rec[((size_t)t * numPorts + p) * 2];
                port.iRec[t] = rec[((size_t)t * numPorts + p) * 2 + 1];
            }
        }
    }
    glFinish();
    msg = GlRendererName();
    return true;
}
