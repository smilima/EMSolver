//---------------------------------------------------------------------------
// GLView.cpp - OpenGL viewport implementation
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "GLView.h"
#include <gl/gl.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>

#pragma package(smart_init)
#pragma comment(lib, "opengl32")

//---------------------------------------------------------------------------
__fastcall TGLView::TGLView(TComponent *Owner) : TCustomControl(Owner)
{
    TabStop = true;
    DoubleBuffered = false;
}

//---------------------------------------------------------------------------
__fastcall TGLView::~TGLView()
{
    releaseContext();
}

//---------------------------------------------------------------------------
void __fastcall TGLView::CreateParams(TCreateParams &Params)
{
    TCustomControl::CreateParams(Params);
    Params.WindowClass.style |= CS_OWNDC;
}

//---------------------------------------------------------------------------
void __fastcall TGLView::WMEraseBkgnd(TMessage &msg)
{
    msg.Result = 1;   // GL repaints everything; avoid flicker
}

//---------------------------------------------------------------------------
void TGLView::ensureContext()
{
    if (hglrc)
        return;
    hdc = GetDC(Handle);
    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize      = sizeof(pfd);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, pf, &pfd);
    hglrc = wglCreateContext(hdc);
}

//---------------------------------------------------------------------------
void TGLView::releaseContext()
{
    if (hglrc)
    {
        wglMakeCurrent(0, 0);
        wglDeleteContext(hglrc);
        hglrc = 0;
        fontBase = fontBaseAxis = 0;   // lists died with the context; rebuild
    }
    if (hdc && HandleAllocated())
    {
        ReleaseDC(Handle, hdc);
        hdc = 0;
    }
}

//---------------------------------------------------------------------------
// scene data updates
//---------------------------------------------------------------------------
void TGLView::setScene(const std::vector<SceneObject> &objs, int selected)
{
    meshes.clear();
    wires.clear();
    feedMarkers.clear();
    for (size_t i = 0; i < objs.size(); ++i)
    {
        const SceneObject &o = objs[i];
        bool sel = ((int)i == selected);
        if (!o.mesh.verts.empty())
        {
            DispMesh dm;
            dm.selected = sel;
            dm.v.reserve(o.mesh.verts.size());
            for (const auto &v : o.mesh.verts)
                dm.v.push_back(v + o.position);
            dm.n   = o.mesh.normals;
            dm.idx = o.mesh.idx;
            meshes.push_back(std::move(dm));
        }
        for (const auto &w : o.wires)
        {
            DispWire dw;
            dw.selected = sel;
            for (const auto &p : w.pts)
                dw.pts.push_back(p + o.position);
            wires.push_back(std::move(dw));
        }
        if (o.feed.enabled)
            feedMarkers.push_back((o.feed.a + o.feed.b) * 0.5f + o.position);
    }
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setProbeMarkers(const std::vector<Vec3> &pts)
{
    probeMarkers = pts;
    Invalidate();
}

//---------------------------------------------------------------------------
GLResult TGLView::exportResult() const
{
    GLResult r;
    r.faces = faces; r.faceVals = faceVals; r.faceDl = faceDl;
    r.triV = triCurV; r.triIdx = triCurIdx; r.triMag = triCurMag;
    r.wirePts = wirePts; r.wireMag = wireMag;
    if (planeAxis >= 0 && !planeVals.empty())
    {
        r.hasPlane = true; r.planeGrid = planeGrid; r.planeAxis = planeAxis;
        r.planeIdx = planeIdx; r.planeN1 = planeN1; r.planeN2 = planeN2;
        r.planeVals = planeVals;
    }
    if (!patU.empty())
    {
        r.hasPattern = true;
        r.pattern.nTheta = patNTheta; r.pattern.nPhi = patNPhi;
        r.pattern.U = patU; r.pattern.uMax = patUMax;
        r.patCenter = patCenter; r.patScale = patScale;
    }
    r.domain = domain; r.domainVisible = domainVisible;
    r.probeMarkers = probeMarkers;
    return r;
}

//---------------------------------------------------------------------------
void TGLView::importResult(const GLResult &r)
{
    clearSurfaceData();
    clearTriCurrents();
    clearWireCurrents();
    clearPlaneData();
    clearPattern();
    clearMeshEdges();
    if (!r.faceVals.empty())
        setSurfaceData(r.faces, r.faceVals, r.faceDl);
    if (!r.triMag.empty())
        setTriCurrents(r.triV, r.triIdx, r.triMag);
    if (!r.wireMag.empty())
        setWireCurrents(r.wirePts, r.wireMag);
    if (r.hasPlane)
        setPlaneData(r.planeGrid, r.planeAxis, r.planeIdx, r.planeN1,
                     r.planeN2, r.planeVals);
    if (r.hasPattern)
        setPattern(r.pattern, r.patCenter, r.patScale);
    setDomain(r.domain, r.domainVisible);
    setProbeMarkers(r.probeMarkers);
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setDomain(const Aabb &box, bool show)
{
    domain = box;
    domainVisible = show && box.valid();
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setSurfaceData(const std::vector<SurfaceFace> &f,
                             const std::vector<float> &values, float dl)
{
    faces    = f;
    faceVals = values;
    faceDl   = dl;
    // Robust color normalization: the feed region carries currents tens of
    // dB above everything else; scaling to the 99.5th percentile keeps
    // weakly illuminated surfaces (reflector plates etc.) visible.
    faceNorm = 0.0f;
    if (!values.empty())
    {
        std::vector<float> tmp = values;
        size_t k = (size_t)(tmp.size() * 0.995);
        if (k >= tmp.size())
            k = tmp.size() - 1;
        std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
        faceNorm = tmp[k];
    }
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::clearSurfaceData()
{
    faces.clear();
    faceVals.clear();
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setPlaneData(const VoxelGridSpec &g, int axis, int index,
                           int n1, int n2, const std::vector<float> &vals)
{
    planeGrid = g;
    planeAxis = axis;
    planeIdx  = index;
    planeN1   = n1;
    planeN2   = n2;
    planeVals = vals;
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::clearPlaneData()
{
    planeAxis = -1;
    planeVals.clear();
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setPattern(const FarFieldData &ff, const Vec3 &center, float scale)
{
    patNTheta = ff.nTheta;
    patNPhi   = ff.nPhi;
    patU      = ff.U;
    patUMax   = ff.uMax;
    patCenter = center;
    patScale  = scale;
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::clearPattern()
{
    patU.clear();
    patNTheta = patNPhi = 0;
    Invalidate();
}

//---------------------------------------------------------------------------
bool TGLView::captureFrame(std::vector<unsigned char> &rgb, int &w, int &h)
{
    if (!HandleAllocated())
        return false;
    capturePending = true;
    Repaint();                  // synchronous paint -> fills captureBuf
    capturePending = false;
    if (captureBuf.empty())
        return false;
    rgb = captureBuf;
    w = captureW;
    h = captureH;
    return true;
}

//---------------------------------------------------------------------------
void TGLView::zoomExtents()
{
    Aabb b;
    for (const auto &m : meshes)
        for (const auto &v : m.v)
            b.grow(v);
    for (const auto &w : wires)
        for (const auto &p : w.pts)
            b.grow(p);
    if (domainVisible)
        b.grow(domain);
    if (!b.valid())
    {
        camTarget = Vec3(0, 0, 0);
        camDist   = 1.0f;
    }
    else
    {
        camTarget = b.center();
        camDist   = std::max(0.001f, b.size().length() * 1.2f);
    }
    Invalidate();
}

//---------------------------------------------------------------------------
// mouse / camera
//---------------------------------------------------------------------------
void __fastcall TGLView::MouseDown(TMouseButton Button, TShiftState Shift, int X, int Y)
{
    SetFocus();
    lastX = X; lastY = Y;
    if (Button == mbLeft)
    {
        // defer rotate / probe-drag until the mouse actually moves, so a
        // click that doesn't drag can be treated as a pick (select)
        leftDown    = true;
        dragStarted = false;
        downX = X; downY = Y;
    }
    else if (Button == mbRight || Button == mbMiddle)
        panning = true;
    TCustomControl::MouseDown(Button, Shift, X, Y);
}

//---------------------------------------------------------------------------
void __fastcall TGLView::MouseMove(TShiftState Shift, int X, int Y)
{
    // promote a held left button to a drag once it moves past a small dead
    // zone; below that it stays a click (handled as a pick on mouse-up)
    if (leftDown && !dragStarted)
    {
        int mdx = X - downX, mdy = Y - downY;
        if (mdx < 0) mdx = -mdx;
        if (mdy < 0) mdy = -mdy;
        if (mdx > 3 || mdy > 3)
        {
            dragStarted = true;
            lastX = X; lastY = Y;            // start the drag delta here
            if (probeDragEnabled) { draggingProbe = true; probeDidMove = false; }
            else                    rotating = true;
        }
    }
    int dx = X - lastX, dy = Y - lastY;
    lastX = X; lastY = Y;
    if (draggingProbe)
    {
        // slide the probe in the plane parallel to the screen (same camera
        // basis as the pan code), scaled so it tracks the cursor at its depth
        float cy = std::cos(camYaw), sy = std::sin(camYaw);
        float cp = std::cos(camPitch), sp = std::sin(camPitch);
        Vec3 eye   = camTarget + Vec3(cp * cy, cp * sy, sp) * camDist;
        Vec3 fwd   = (camTarget - eye).normalized();
        Vec3 right = fwd.cross(Vec3(0, 0, 1)).normalized();
        if (right.length2() < 1e-10f)
            right = Vec3(0, 1, 0);
        Vec3 up    = right.cross(fwd);
        int  h     = std::max(1, (int)ClientHeight);
        float depth = std::max(1e-4f, (probeDragPos - eye).dot(fwd));
        float wpp = 2.0f * depth * std::tan(22.5f * (float)M_PI / 180.0f)
                    / (float)h;
        probeDragPos = probeDragPos + right * (dx * wpp) - up * (dy * wpp);
        probeDidMove = true;
        if (onProbeMoved)
            onProbeMoved(probeDragPos, false);
        Invalidate();
        TCustomControl::MouseMove(Shift, X, Y);
        return;
    }
    if (rotating)
    {
        camYaw   -= dx * 0.008f;
        camPitch += dy * 0.008f;
        const float lim = 1.55f;
        if (camPitch >  lim) camPitch =  lim;
        if (camPitch < -lim) camPitch = -lim;
        Invalidate();
    }
    else if (panning)
    {
        // move target in the screen plane
        float cy = std::cos(camYaw), sy = std::sin(camYaw);
        float cp = std::cos(camPitch), sp = std::sin(camPitch);
        Vec3 dir(-cp * cy, -cp * sy, -sp);          // view direction
        Vec3 right = dir.cross(Vec3(0, 0, 1)).normalized();
        Vec3 up    = right.cross(dir).normalized();
        float k = camDist * 0.0016f;
        camTarget -= right * (dx * k);
        camTarget += up * (dy * k);
        Invalidate();
    }
    TCustomControl::MouseMove(Shift, X, Y);
}

//---------------------------------------------------------------------------
void __fastcall TGLView::MouseUp(TMouseButton Button, TShiftState Shift, int X, int Y)
{
    if (Button == mbLeft)
    {
        if (leftDown && !dragStarted && onPick)
        {
            // a click (no drag): pick the object under the cursor
            Vec3 ro, rd;
            if (computeRay(downX, downY, ro, rd))
                onPick(ro, rd);
        }
        if (draggingProbe)
        {
            draggingProbe = false;
            if (probeDidMove && onProbeMoved)
                onProbeMoved(probeDragPos, true);   // final: commit / mark stale
            probeDidMove = false;
        }
        leftDown = dragStarted = rotating = false;
    }
    else if (Button == mbRight || Button == mbMiddle)
        panning = false;
    TCustomControl::MouseUp(Button, Shift, X, Y);
}

//---------------------------------------------------------------------------
// World-space pick ray (origin at the eye) through screen pixel (px, py).
bool TGLView::computeRay(int px, int py, Vec3 &origin, Vec3 &dir)
{
    int w = std::max(1, (int)ClientWidth), h = std::max(1, (int)ClientHeight);
    float cy = std::cos(camYaw), sy = std::sin(camYaw);
    float cp = std::cos(camPitch), sp = std::sin(camPitch);
    Vec3 eye   = camTarget + Vec3(cp * cy, cp * sy, sp) * camDist;
    Vec3 fwd   = (camTarget - eye).normalized();
    Vec3 right = fwd.cross(Vec3(0, 0, 1)).normalized();
    if (right.length2() < 1e-10f)
        right = Vec3(0, 1, 0);
    Vec3 up    = right.cross(fwd);
    float aspect = (float)w / (float)h;
    float tan22  = (float)std::tan(22.5 * M_PI / 180.0);
    float ndcX = 2.0f * (float)px / (float)w - 1.0f;
    float ndcY = 1.0f - 2.0f * (float)py / (float)h;
    origin = eye;
    dir = (fwd + right * (ndcX * aspect * tan22) + up * (ndcY * tan22))
          .normalized();
    return true;
}

//---------------------------------------------------------------------------
void TGLView::setProbeDrag(bool enabled, const Vec3 &pos)
{
    probeDragEnabled = enabled;
    probeDragPos = pos;
    if (!enabled)
        draggingProbe = false;
}

//---------------------------------------------------------------------------
bool __fastcall TGLView::DoMouseWheel(TShiftState Shift, int WheelDelta,
                                      const System::Types::TPoint &MousePos)
{
    camDist *= std::pow(0.999f, (float)WheelDelta);
    if (camDist < 1e-4f) camDist = 1e-4f;
    Invalidate();
    return true;
}

//---------------------------------------------------------------------------
// rendering
//---------------------------------------------------------------------------
static void JetColor(float t, float rgb[3])
{
    t = std::max(0.0f, std::min(1.0f, t));
    rgb[0] = std::max(0.0f, std::min(1.0f, 1.5f - std::fabs(4.0f * t - 3.0f)));
    rgb[1] = std::max(0.0f, std::min(1.0f, 1.5f - std::fabs(4.0f * t - 2.0f)));
    rgb[2] = std::max(0.0f, std::min(1.0f, 1.5f - std::fabs(4.0f * t - 1.0f)));
}

//---------------------------------------------------------------------------
void TGLView::applyCamera()
{
    int w = std::max(1, (int)ClientWidth), h = std::max(1, (int)ClientHeight);
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (double)w / h;
    double zn = camDist * 0.01, zf = camDist * 50.0;
    double t = zn * std::tan(22.5 * M_PI / 180.0);
    glFrustum(-t * aspect, t * aspect, -t, t, zn, zf);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float cy = std::cos(camYaw), sy = std::sin(camYaw);
    float cp = std::cos(camPitch), sp = std::sin(camPitch);
    Vec3 eye = camTarget + Vec3(cp * cy, cp * sy, sp) * camDist;
    Vec3 f = (camTarget - eye).normalized();
    Vec3 s = f.cross(Vec3(0, 0, 1)).normalized();
    if (s.length2() < 1e-10f)
        s = Vec3(0, 1, 0);
    Vec3 u = s.cross(f);
    float m[16] = {
        s.x, u.x, -f.x, 0,
        s.y, u.y, -f.y, 0,
        s.z, u.z, -f.z, 0,
        0,   0,    0,   1 };
    glMultMatrixf(m);
    glTranslatef(-eye.x, -eye.y, -eye.z);
}

//---------------------------------------------------------------------------
void __fastcall TGLView::Paint()
{
    ensureContext();
    if (!hglrc)
        return;
    wglMakeCurrent(hdc, hglrc);

    glClearColor(0.13f, 0.15f, 0.18f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    applyCamera();
    drawScene();
    drawAxes();
    drawLegend();

    if (capturePending)
    {
        int w = std::max(1, (int)ClientWidth), h = std::max(1, (int)ClientHeight);
        captureW = w;
        captureH = h;
        captureBuf.assign((size_t)w * h * 3, 0);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadBuffer(GL_BACK);
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, captureBuf.data());
        // GL rows are bottom-up; flip to top-down
        std::vector<unsigned char> row((size_t)w * 3);
        for (int y = 0; y < h / 2; ++y)
        {
            unsigned char *a = &captureBuf[(size_t)y * w * 3];
            unsigned char *b = &captureBuf[(size_t)(h - 1 - y) * w * 3];
            memcpy(row.data(), a, row.size());
            memcpy(a, b, row.size());
            memcpy(b, row.data(), row.size());
        }
    }

    SwapBuffers(hdc);
}

//---------------------------------------------------------------------------
void TGLView::drawScene()
{
    // lighting for meshes
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1);
    float lpos[4] = { 0.4f, 0.3f, 0.85f, 0.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, lpos);
    float amb[4] = { 0.35f, 0.35f, 0.35f, 1.0f };
    glLightfv(GL_LIGHT0, GL_AMBIENT, amb);

    if (showModel)
        drawMeshes();
    glDisable(GL_LIGHTING);
    if (showModel)
        drawWires();

    // feed markers
    glPointSize(9.0f);
    glColor3f(1.0f, 0.25f, 0.2f);
    glBegin(GL_POINTS);
    for (const auto &p : feedMarkers)
        glVertex3f(p.x, p.y, p.z);
    glEnd();

    // E-field probe markers (cyan crosshair)
    if (!probeMarkers.empty())
    {
        glColor3f(0.2f, 0.95f, 1.0f);
        glPointSize(11.0f);
        glBegin(GL_POINTS);
        for (const auto &p : probeMarkers)
            glVertex3f(p.x, p.y, p.z);
        glEnd();
        glLineWidth(1.5f);
        glBegin(GL_LINES);
        for (const auto &p : probeMarkers)
        {
            // small 3D crosshair so it is visible against geometry
            float s = 0.04f * camDist;
            glVertex3f(p.x - s, p.y, p.z); glVertex3f(p.x + s, p.y, p.z);
            glVertex3f(p.x, p.y - s, p.z); glVertex3f(p.x, p.y + s, p.z);
            glVertex3f(p.x, p.y, p.z - s); glVertex3f(p.x, p.y, p.z + s);
        }
        glEnd();
        glLineWidth(1.0f);
    }

    drawSurfaceFaces();
    if (domainVisible)
        drawDomainBox();
    drawPattern();
    drawPlaneWaveMarker();

    // FEM mesh wireframe
    if (!meshSegs.empty())
    {
        glDisable(GL_LIGHTING);
        glColor3f(0.45f, 0.62f, 0.75f);
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        for (const auto &p : meshSegs)
            glVertex3f(p.x, p.y, p.z);
        glEnd();
    }

    // MoM wire currents (color each segment by |I|)
    if (!wirePts.empty() && wireMag.size() * 2 == wirePts.size())
    {
        float norm = (wireNorm > 0.0f) ? wireNorm : 1.0f;
        currentMax = 0.0f;
        for (float v : wireMag)
            currentMax = std::max(currentMax, v);
        glDisable(GL_LIGHTING);
        glLineWidth(4.0f);
        glBegin(GL_LINES);
        for (size_t s = 0; s < wireMag.size(); ++s)
        {
            float t;
            if (dbScale)
            {
                float db = 20.0f * std::log10(std::max(wireMag[s],
                              norm * 1e-7f) / norm);
                t = (db + dbRange) / dbRange;
            }
            else
                t = wireMag[s] / norm;
            if (t > 1.0f) t = 1.0f;
            float rgb[3];
            JetColor(t, rgb);
            glColor3f(rgb[0], rgb[1], rgb[2]);
            const Vec3 &a = wirePts[s * 2], &b = wirePts[s * 2 + 1];
            glVertex3f(a.x, a.y, a.z);
            glVertex3f(b.x, b.y, b.z);
        }
        glEnd();
        glLineWidth(1.0f);
    }

    // MoM surface currents (color each triangle by |J|)
    if (!triCurIdx.empty() && triCurMag.size() * 3 == triCurIdx.size())
    {
        float norm = (triCurNorm > 0.0f) ? triCurNorm : 1.0f;
        currentMax = 0.0f;
        for (float v : triCurMag)
            currentMax = std::max(currentMax, v);
        glDisable(GL_LIGHTING);
        glBegin(GL_TRIANGLES);
        for (size_t t = 0; t < triCurMag.size(); ++t)
        {
            float tt;
            if (dbScale)
            {
                float db = 20.0f * std::log10(std::max(triCurMag[t],
                              norm * 1e-7f) / norm);
                tt = (db + dbRange) / dbRange;
            }
            else
                tt = triCurMag[t] / norm;
            if (tt > 1.0f) tt = 1.0f;
            float rgb[3];
            JetColor(tt, rgb);
            glColor3f(rgb[0], rgb[1], rgb[2]);
            for (int c = 0; c < 3; ++c)
            {
                const Vec3 &v = triCurV[triCurIdx[t * 3 + c]];
                glVertex3f(v.x, v.y, v.z);
            }
        }
        glEnd();
    }

    drawFieldPlane();
}

//---------------------------------------------------------------------------
void TGLView::setPlaneWaveMarker(const Aabb &domain, int propAxis, int polAxis,
                                 bool show)
{
    pwDomain = domain;
    pwAxis   = propAxis;
    pwPol    = polAxis;
    pwShow   = show && domain.valid();
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::setMeshEdges(const std::vector<Vec3> &segments)
{
    meshSegs = segments;
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::clearMeshEdges()
{
    meshSegs.clear();
    Invalidate();
}

void TGLView::setWireCurrents(const std::vector<Vec3> &pts,
                              const std::vector<float> &mag)
{
    wirePts = pts;
    wireMag = mag;
    wireNorm = 0.0f;
    if (!mag.empty())
    {
        std::vector<float> tmp = mag;
        size_t kk = (size_t)(tmp.size() * 0.99);
        if (kk >= tmp.size()) kk = tmp.size() - 1;
        std::nth_element(tmp.begin(), tmp.begin() + kk, tmp.end());
        wireNorm = tmp[kk];
    }
    Invalidate();
}

void TGLView::clearWireCurrents()
{
    wirePts.clear();
    wireMag.clear();
    Invalidate();
}

void TGLView::setTriCurrents(const std::vector<Vec3> &verts,
                             const std::vector<int> &idx,
                             const std::vector<float> &triMag)
{
    triCurV = verts;
    triCurIdx = idx;
    triCurMag = triMag;
    triCurNorm = 0.0f;
    if (!triMag.empty())
    {
        std::vector<float> tmp = triMag;
        size_t kk = (size_t)(tmp.size() * 0.99);
        if (kk >= tmp.size()) kk = tmp.size() - 1;
        std::nth_element(tmp.begin(), tmp.begin() + kk, tmp.end());
        triCurNorm = tmp[kk];
    }
    Invalidate();
}

void TGLView::clearTriCurrents()
{
    triCurV.clear();
    triCurIdx.clear();
    triCurMag.clear();
    Invalidate();
}

//---------------------------------------------------------------------------
void TGLView::drawPlaneWaveMarker()
{
    if (!pwShow || !pwDomain.valid())
        return;
    const int a  = pwAxis;
    const int t1 = (a == 0) ? 1 : 0;
    const int t2 = (a == 2) ? 1 : 2;
    Vec3 sz = pwDomain.size();
    Vec3 c  = pwDomain.center();
    float faceMin = std::min(sz[t1], sz[t2]);
    float half  = 0.14f * faceMin;        // horn aperture half-size
    float wg    = 0.35f * half;           // waveguide half-size
    float len   = 3.0f * half;            // flare length
    float wgLen = 1.2f * half;            // waveguide stub length
    float gap   = 0.04f * faceMin;

    // aperture just outside the wave-entry (low) face, boresight +axis
    Vec3 apC = c;
    apC.set(a, pwDomain.lo[a] - gap);
    Vec3 wgC = apC;
    wgC.set(a, apC[a] - len);
    Vec3 bkC = wgC;
    bkC.set(a, wgC[a] - wgLen);

    auto at = [&](const Vec3 &base, float d1, float d2)
    {
        Vec3 p = base;
        p.set(t1, base[t1] + d1);
        p.set(t2, base[t2] + d2);
        return p;
    };
    Vec3 A1 = at(apC, -half, -half), A2 = at(apC, half, -half);
    Vec3 A3 = at(apC,  half,  half), A4 = at(apC, -half, half);
    Vec3 W1 = at(wgC, -wg, -wg), W2 = at(wgC, wg, -wg);
    Vec3 W3 = at(wgC,  wg,  wg), W4 = at(wgC, -wg, wg);
    Vec3 B1 = at(bkC, -wg, -wg), B2 = at(bkC, wg, -wg);
    Vec3 B3 = at(bkC,  wg,  wg), B4 = at(bkC, -wg, wg);

    glDisable(GL_LIGHTING);

    // translucent flare walls
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glColor4f(1.0f, 0.62f, 0.18f, 0.30f);
    glBegin(GL_QUADS);
    auto q4 = [](const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, const Vec3 &p4)
    {
        glVertex3f(p1.x, p1.y, p1.z); glVertex3f(p2.x, p2.y, p2.z);
        glVertex3f(p3.x, p3.y, p3.z); glVertex3f(p4.x, p4.y, p4.z);
    };
    q4(A1, A2, W2, W1);
    q4(A2, A3, W3, W2);
    q4(A3, A4, W4, W3);
    q4(A4, A1, W1, W4);
    glEnd();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    // edges: aperture, flare, waveguide stub
    glColor3f(1.0f, 0.72f, 0.25f);
    glLineWidth(2.0f);
    glBegin(GL_LINE_LOOP);
    glVertex3f(A1.x, A1.y, A1.z); glVertex3f(A2.x, A2.y, A2.z);
    glVertex3f(A3.x, A3.y, A3.z); glVertex3f(A4.x, A4.y, A4.z);
    glEnd();
    glBegin(GL_LINES);
    auto seg = [](const Vec3 &p, const Vec3 &q)
    {
        glVertex3f(p.x, p.y, p.z);
        glVertex3f(q.x, q.y, q.z);
    };
    seg(A1, W1); seg(A2, W2); seg(A3, W3); seg(A4, W4);
    seg(W1, B1); seg(W2, B2); seg(W3, B3); seg(W4, B4);
    seg(B1, B2); seg(B2, B3); seg(B3, B4); seg(B4, B1);
    seg(W1, W2); seg(W2, W3); seg(W3, W4); seg(W4, W1);
    glEnd();

    // propagation arrow (into the domain, along +axis)
    Vec3 ar0 = c, ar1 = c;
    ar0.set(a, pwDomain.lo[a] + 0.03f * sz[a]);
    ar1.set(a, ar0[a] + 2.6f * half);
    glColor3f(0.35f, 1.0f, 0.45f);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    seg(ar0, ar1);
    float hd = 0.45f * half;
    Vec3 hb = ar1;
    hb.set(a, ar1[a] - hd);
    seg(ar1, at(hb,  hd * 0.6f, 0));
    seg(ar1, at(hb, -hd * 0.6f, 0));
    seg(ar1, at(hb, 0,  hd * 0.6f));
    seg(ar1, at(hb, 0, -hd * 0.6f));
    glEnd();

    // E-field polarization: double-headed line at the arrow midpoint
    Vec3 pm = c;
    pm.set(a, (ar0[a] + ar1[a]) * 0.5f);
    Vec3 e0 = pm, e1 = pm;
    e0.set(pwPol, pm[pwPol] - 0.9f * half);
    e1.set(pwPol, pm[pwPol] + 0.9f * half);
    glColor3f(1.0f, 0.35f, 0.45f);
    glBegin(GL_LINES);
    seg(e0, e1);
    int pt = (pwPol == a) ? t1 : ((pwPol == t1) ? t2 : t1);
    auto eHead = [&](const Vec3 &tip, float dir)
    {
        Vec3 b = tip;
        b.set(pwPol, tip[pwPol] - dir * 0.3f * half);
        Vec3 h1 = b, h2 = b;
        h1.set(pt, b[pt] + 0.18f * half);
        h2.set(pt, b[pt] - 0.18f * half);
        seg(tip, h1);
        seg(tip, h2);
    };
    eHead(e1, +1.0f);
    eHead(e0, -1.0f);
    glEnd();
    glLineWidth(1.0f);
}

//---------------------------------------------------------------------------
void TGLView::drawPattern()
{
    if (patU.empty() || patNTheta < 2 || patNPhi < 2 || patUMax <= 0.0f)
        return;
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto vert = [&](int ti, int pi)
    {
        float th = (float)M_PI * ti / (patNTheta - 1);
        float ph = 2.0f * (float)M_PI * pi / (patNPhi - 1);
        float u = patU[(size_t)ti * patNPhi + pi];
        // 30 dB dynamic range mapped to radius 0.03..1
        float db = 10.0f * std::log10(std::max(u, patUMax * 1e-7f) / patUMax);
        float t = std::max(0.0f, (db + 30.0f) / 30.0f);
        float r = (0.03f + 0.97f * t) * patScale;
        float rgb[3];
        JetColor(t, rgb);
        glColor4f(rgb[0], rgb[1], rgb[2], 0.65f);
        float st = std::sin(th), ct = std::cos(th);
        glVertex3f(patCenter.x + r * st * std::cos(ph),
                   patCenter.y + r * st * std::sin(ph),
                   patCenter.z + r * ct);
    };
    for (int ti = 0; ti + 1 < patNTheta; ++ti)
    {
        glBegin(GL_QUAD_STRIP);
        for (int pi = 0; pi < patNPhi; ++pi)
        {
            vert(ti, pi);
            vert(ti + 1, pi);
        }
        glEnd();
    }
    glDisable(GL_BLEND);
}

//---------------------------------------------------------------------------
void TGLView::drawMeshes()
{
    bool currentsActive = !faceVals.empty();
    for (const auto &m : meshes)
    {
        if (m.selected)
            glColor3f(0.95f, 0.75f, 0.3f);
        else if (currentsActive)
            glColor3f(0.25f, 0.27f, 0.3f);
        else
            glColor3f(0.62f, 0.66f, 0.72f);
        glBegin(GL_TRIANGLES);
        for (size_t i = 0; i + 2 < m.idx.size(); i += 3)
            for (int q = 0; q < 3; ++q)
            {
                int vi = m.idx[i + q];
                if (vi < (int)m.n.size())
                {
                    const Vec3 &n = m.n[vi];
                    glNormal3f(n.x, n.y, n.z);
                }
                const Vec3 &v = m.v[vi];
                glVertex3f(v.x, v.y, v.z);
            }
        glEnd();
    }
}

//---------------------------------------------------------------------------
void TGLView::drawWires()
{
    glLineWidth(2.5f);
    for (const auto &w : wires)
    {
        if (w.selected)
            glColor3f(1.0f, 0.85f, 0.35f);
        else
            glColor3f(0.85f, 0.62f, 0.25f);
        glBegin(GL_LINE_STRIP);
        for (const auto &p : w.pts)
            glVertex3f(p.x, p.y, p.z);
        glEnd();
    }
    glLineWidth(1.0f);
}

//---------------------------------------------------------------------------
void TGLView::drawSurfaceFaces()
{
    if (faces.empty() || faceVals.size() != faces.size())
    {
        currentMax = 0.0f;
        return;
    }
    float vmaxv = 0.0f;
    for (float v : faceVals)
        vmaxv = std::max(vmaxv, v);
    currentMax = vmaxv;
    if (vmaxv <= 0.0f)
        return;
    float norm = (faceNorm > 0.0f) ? faceNorm : vmaxv;

    const float h = faceDl * 0.5f;
    glDisable(GL_LIGHTING);
    glBegin(GL_QUADS);
    for (size_t i = 0; i < faces.size(); ++i)
    {
        const SurfaceFace &f = faces[i];
        float t;
        if (dbScale)
        {
            float db = 20.0f * std::log10(std::max(faceVals[i], norm * 1e-7f) / norm);
            t = (db + dbRange) / dbRange;
        }
        else
            t = faceVals[i] / norm;
        if (t > 1.0f)
            t = 1.0f;
        float rgb[3];
        JetColor(t, rgb);
        glColor3f(rgb[0], rgb[1], rgb[2]);

        // quad in the plane normal to f.axis, nudged toward the air side
        Vec3 c = f.center;
        c.set(f.axis, c[f.axis] + f.sign * faceDl * 0.02f);
        int a1 = (f.axis == 0) ? 1 : 0;
        int a2 = (f.axis == 2) ? 1 : 2;
        Vec3 e1(0, 0, 0), e2(0, 0, 0);
        e1.set(a1, h);
        e2.set(a2, h);
        Vec3 p1 = c - e1 - e2, p2 = c + e1 - e2, p3 = c + e1 + e2, p4 = c - e1 + e2;
        glVertex3f(p1.x, p1.y, p1.z);
        glVertex3f(p2.x, p2.y, p2.z);
        glVertex3f(p3.x, p3.y, p3.z);
        glVertex3f(p4.x, p4.y, p4.z);
    }
    glEnd();
}

//---------------------------------------------------------------------------
void TGLView::drawFieldPlane()
{
    if (planeAxis < 0 || planeVals.empty())
        return;
    float vmaxv = 0.0f;
    for (float v : planeVals)
        vmaxv = std::max(vmaxv, v);
    if (vmaxv <= 0.0f)
        return;

    int a1 = (planeAxis == 0) ? 1 : 0;
    int a2 = (planeAxis == 2) ? 1 : 2;
    float dl = planeGrid.dl;
    float pc = planeGrid.origin[planeAxis] + (planeIdx + 0.5f) * dl;

    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glBegin(GL_QUADS);
    for (int q2 = 0; q2 < planeN2; ++q2)
        for (int q1 = 0; q1 < planeN1; ++q1)
        {
            float v = planeVals[(size_t)q2 * planeN1 + q1];
            float db = 20.0f * std::log10(std::max(v, vmaxv * 1e-6f) / vmaxv);
            float t = (db + 40.0f) / 40.0f;
            float rgb[3];
            JetColor(t, rgb);
            glColor4f(rgb[0], rgb[1], rgb[2], 0.55f);
            Vec3 p;
            p.set(planeAxis, pc);
            float c1 = planeGrid.origin[a1] + q1 * dl;
            float c2 = planeGrid.origin[a2] + q2 * dl;
            p.set(a1, c1);      p.set(a2, c2);      glVertex3f(p.x, p.y, p.z);
            p.set(a1, c1 + dl);                     glVertex3f(p.x, p.y, p.z);
            p.set(a2, c2 + dl);                     glVertex3f(p.x, p.y, p.z);
            p.set(a1, c1);                          glVertex3f(p.x, p.y, p.z);
        }
    glEnd();
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

//---------------------------------------------------------------------------
void TGLView::drawDomainBox()
{
    glDisable(GL_LIGHTING);
    glColor3f(0.45f, 0.5f, 0.55f);
    glLineWidth(1.0f);
    const Vec3 &l = domain.lo, &h = domain.hi;
    Vec3 c[8] = {
        {l.x,l.y,l.z},{h.x,l.y,l.z},{h.x,h.y,l.z},{l.x,h.y,l.z},
        {l.x,l.y,h.z},{h.x,l.y,h.z},{h.x,h.y,h.z},{l.x,h.y,h.z} };
    static const int e[12][2] = {
        {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7} };
    glBegin(GL_LINES);
    for (auto &ed : e)
    {
        glVertex3f(c[ed[0]].x, c[ed[0]].y, c[ed[0]].z);
        glVertex3f(c[ed[1]].x, c[ed[1]].y, c[ed[1]].z);
    }
    glEnd();
}

//---------------------------------------------------------------------------
void TGLView::drawAxes()
{
    // triad in the lower-left corner, sized to keep the X/Y/Z labels legible
    glViewport(8, 8, 128, 128);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.8, 1.8, -1.8, 1.8, -2.5, 2.5);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    float cy = std::cos(camYaw), sy = std::sin(camYaw);
    float cp = std::cos(camPitch), sp = std::sin(camPitch);
    Vec3 eye(cp * cy, cp * sy, sp);
    Vec3 f = (-eye).normalized();
    Vec3 s = f.cross(Vec3(0, 0, 1)).normalized();
    if (s.length2() < 1e-10f)
        s = Vec3(0, 1, 0);
    Vec3 u = s.cross(f);
    float m[16] = {
        s.x, u.x, -f.x, 0,
        s.y, u.y, -f.y, 0,
        s.z, u.z, -f.z, 0,
        0,   0,    0,   1 };
    glMultMatrixf(m);
    glTranslatef(-eye.x, -eye.y, -eye.z);

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glColor3f(0.9f, 0.3f, 0.3f); glVertex3f(0, 0, 0); glVertex3f(1, 0, 0);
    glColor3f(0.3f, 0.9f, 0.3f); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0);
    glColor3f(0.35f, 0.5f, 1.0f); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1);
    glEnd();
    glLineWidth(1.0f);

    // axis letters, just past each tip, colour-matched to the axes
    ensureFont();
    glColor3f(1.0f, 0.55f, 0.55f); drawText3(1.12f, 0.0f, 0.0f, "X");
    glColor3f(0.55f, 1.0f, 0.55f); drawText3(0.0f, 1.12f, 0.0f, "Y");
    glColor3f(0.6f,  0.7f,  1.0f); drawText3(0.0f, 0.0f, 1.12f, "Z");

    glEnable(GL_DEPTH_TEST);
}

//---------------------------------------------------------------------------
// Lazily build a bitmap font (display lists) for 2D overlay text. Requires a
// current GL context, so it is called from the draw path, not setup.
void TGLView::ensureFont()
{
    if (fontBase || !hdc)
        return;
    LOGFONTA lf;
    memset(&lf, 0, sizeof(lf));
    lf.lfCharSet = ANSI_CHARSET;
    lf.lfQuality = ANTIALIASED_QUALITY;
    strcpy(lf.lfFaceName, "Segoe UI");

    // small font for the legend text
    fontBase = glGenLists(96);              // ASCII 32..127
    lf.lfHeight = -13;
    lf.lfWeight = FW_NORMAL;
    HFONT font = CreateFontIndirectA(&lf);
    HFONT prev = (HFONT)SelectObject(hdc, font);
    wglUseFontBitmapsA(hdc, 32, 96, fontBase);
    SelectObject(hdc, prev);
    DeleteObject(font);

    // larger bold font for the axis X/Y/Z labels
    fontBaseAxis = glGenLists(96);
    lf.lfHeight = -20;
    lf.lfWeight = FW_BOLD;
    HFONT fontA = CreateFontIndirectA(&lf);
    prev = (HFONT)SelectObject(hdc, fontA);
    wglUseFontBitmapsA(hdc, 32, 96, fontBaseAxis);
    SelectObject(hdc, prev);
    DeleteObject(fontA);
}

//---------------------------------------------------------------------------
// Draw a string with its lower-left at screen (x, y) in the current 2D ortho.
void TGLView::drawText(int x, int y, const char *s)
{
    if (!fontBase || !s || !*s)
        return;
    glRasterPos2i(x, y);
    glListBase(fontBase - 32);
    glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, s);
}

//---------------------------------------------------------------------------
// Draw a string anchored at world point (x, y, z) under the current 3D
// transform (the bitmap is laid out in screen pixels from that point).
void TGLView::drawText3(float x, float y, float z, const char *s)
{
    if (!fontBaseAxis || !s || !*s)
        return;
    glRasterPos3f(x, y, z);
    glListBase(fontBaseAxis - 32);
    glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, s);
}

//---------------------------------------------------------------------------
void TGLView::drawLegend()
{
    if (faceVals.empty())
        return;
    int w = std::max(1, (int)ClientWidth), h = std::max(1, (int)ClientHeight);
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    int x0 = w - 34, x1 = w - 14, y0 = h / 2 - 110, y1 = h / 2 + 110;
    glBegin(GL_QUAD_STRIP);
    const int bands = 32;
    for (int i = 0; i <= bands; ++i)
    {
        float t = (float)i / bands;
        float rgb[3];
        JetColor(t, rgb);
        glColor3f(rgb[0], rgb[1], rgb[2]);
        float y = y0 + t * (y1 - y0);
        glVertex2f((float)x0, y);
        glVertex2f((float)x1, y);
    }
    glEnd();
    glColor3f(0.8f, 0.8f, 0.8f);
    glBegin(GL_LINE_LOOP);
    glVertex2f((float)x0, (float)y0);
    glVertex2f((float)x1, (float)y0);
    glVertex2f((float)x1, (float)y1);
    glVertex2f((float)x0, (float)y1);
    glEnd();

    // ---- range + unit labels --------------------------------------------
    ensureFont();
    float norm = (faceNorm > 0.0f) ? faceNorm : currentMax;   // value at t=1
    int   ymid = (y0 + y1) / 2;
    int   lx   = x0 - 6;                  // labels sit just left of the bar
    char  buf[48];
    glColor3f(0.90f, 0.92f, 0.96f);

    if (dbScale)
    {
        // bar spans 0 dB (top) down to -dbRange (bottom), relative to 'norm'
        drawText(lx - 26, y1 - 5, "0 dB");
        snprintf(buf, sizeof(buf), "%g", -0.5 * dbRange);
        drawText(lx - 36, ymid - 5, buf);
        snprintf(buf, sizeof(buf), "%g", -(double)dbRange);
        drawText(lx - 42, y0 - 5, buf);
        // header: quantity + scale; reference line gives the absolute value
        drawText(x0 - 60, y1 + 16, "|Js| (dB)");
        snprintf(buf, sizeof(buf), "0 dB = %.3g A/m", (double)norm);
        drawText(x0 - 96, y0 - 22, buf);
    }
    else
    {
        snprintf(buf, sizeof(buf), "%.3g", (double)norm);
        drawText(lx - 48, y1 - 5, buf);
        snprintf(buf, sizeof(buf), "%.3g", 0.5 * (double)norm);
        drawText(lx - 48, ymid - 5, buf);
        drawText(lx - 12, y0 - 5, "0");
        drawText(x0 - 78, y1 + 16, "|Js| (A/m)");
    }

    glEnable(GL_DEPTH_TEST);
}
