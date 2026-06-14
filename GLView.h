//---------------------------------------------------------------------------
// GLView.h - OpenGL 3D viewport as a VCL control (orbit/pan/zoom camera)
//---------------------------------------------------------------------------
#ifndef GLViewH
#define GLViewH

#include <vcl.h>
#include <Vcl.Controls.hpp>
#include "Geometry.h"
#include "TlmSolver.h"
#include <vector>

// Snapshot of everything shown in the viewport, for project save/load.
struct GLResult
{
    std::vector<SurfaceFace> faces;
    std::vector<float>       faceVals;
    float                    faceDl = 0.0f;
    std::vector<Vec3>        triV;
    std::vector<int>         triIdx;
    std::vector<float>       triMag;
    std::vector<Vec3>        wirePts;
    std::vector<float>       wireMag;
    bool                     hasPlane = false;
    VoxelGridSpec            planeGrid;
    int                      planeAxis = -1, planeIdx = 0, planeN1 = 0, planeN2 = 0;
    std::vector<float>       planeVals;
    bool                     hasPattern = false;
    FarFieldData             pattern;
    Vec3                     patCenter;
    float                    patScale = 1.0f;
    Aabb                     domain;
    bool                     domainVisible = false;
    std::vector<Vec3>        probeMarkers;
};

class TGLView : public TCustomControl
{
public:
    __fastcall TGLView(TComponent *Owner);
    __fastcall virtual ~TGLView();

    GLResult exportResult() const;     // capture current display for saving
    void     importResult(const GLResult &r);  // restore a saved display

    // scene content ---------------------------------------------------------
    void setScene(const std::vector<SceneObject> &objs, int selected);
    void setDomain(const Aabb &box, bool show);
    // surface current overlay: values per face (empty -> hide)
    void setSurfaceData(const std::vector<SurfaceFace> &faces,
                        const std::vector<float> &values, float dl);
    void clearSurfaceData();
    // |E| cut plane
    void setPlaneData(const VoxelGridSpec &g, int axis, int index,
                      int n1, int n2, const std::vector<float> &vals);
    void clearPlaneData();

    // 3D far-field pattern (radius ~ dB-scaled radiation intensity)
    void setPattern(const FarFieldData &ff, const Vec3 &center, float scale);
    void clearPattern();
    bool hasPattern() const { return !patU.empty(); }

    // plane-wave source glyph: horn outside the domain box on the entry
    // face, propagation arrow into the domain, E-polarization indicator
    void setPlaneWaveMarker(const Aabb &domain, int propAxis, int polAxis,
                            bool show);

    // FEM mesh wireframe (point pairs)
    void setMeshEdges(const std::vector<Vec3> &segments);
    void clearMeshEdges();

    // MoM wire currents: 'pts' = 2 points per segment, 'mag' = |I| per segment
    void setWireCurrents(const std::vector<Vec3> &pts,
                         const std::vector<float> &mag);
    void clearWireCurrents();

    // MoM surface currents: vertices, triangle indices (3/tri), |J| per tri
    void setTriCurrents(const std::vector<Vec3> &verts,
                        const std::vector<int> &idx,
                        const std::vector<float> &triMag);
    void clearTriCurrents();

    // E-field probe markers (world positions)
    void setProbeMarkers(const std::vector<Vec3> &pts);

    // synchronous framebuffer capture (RGB, top-down row order)
    bool captureFrame(std::vector<unsigned char> &rgb, int &w, int &h);

    void zoomExtents();

    bool  showModel    = true;
    bool  dbScale      = true;     // color scale: log (dbRange), else linear
    float dbRange      = 40.0f;    // dynamic range of the color map in dB
    float currentMax   = 0.0f;     // read back by UI for the legend

protected:
    virtual void __fastcall CreateParams(TCreateParams &Params);
    virtual void __fastcall Paint();
    void __fastcall WMEraseBkgnd(TMessage &msg);
    DYNAMIC void __fastcall MouseDown(TMouseButton Button, TShiftState Shift, int X, int Y);
    DYNAMIC void __fastcall MouseMove(TShiftState Shift, int X, int Y);
    DYNAMIC void __fastcall MouseUp(TMouseButton Button, TShiftState Shift, int X, int Y);
    DYNAMIC bool __fastcall DoMouseWheel(TShiftState Shift, int WheelDelta, const System::Types::TPoint &MousePos);

    BEGIN_MESSAGE_MAP
        VCL_MESSAGE_HANDLER(WM_ERASEBKGND, TMessage, WMEraseBkgnd)
    END_MESSAGE_MAP(TCustomControl)

private:
    void ensureContext();
    void releaseContext();
    void applyCamera();
    void drawScene();
    void drawMeshes();
    void drawWires();
    void drawSurfaceFaces();
    void drawFieldPlane();
    void drawPattern();
    void drawPlaneWaveMarker();
    void drawDomainBox();
    void drawAxes();
    void drawLegend();

    HDC   hdc  = 0;
    HGLRC hglrc = 0;

    // camera
    Vec3  camTarget{0, 0, 0};
    float camDist  = 1.0f;
    float camYaw   = 0.6f;     // radians
    float camPitch = 0.45f;
    int   lastX = 0, lastY = 0;
    bool  rotating = false, panning = false;

    // display copies (owned by this control, updated from UI thread)
    struct DispMesh
    {
        std::vector<Vec3> v, n;
        std::vector<int>  idx;
        bool selected;
    };
    struct DispWire
    {
        std::vector<Vec3> pts;
        bool selected;
    };
    std::vector<DispMesh> meshes;
    std::vector<DispWire> wires;
    std::vector<Vec3>     feedMarkers;
    std::vector<Vec3>     probeMarkers;

    Aabb  domain;
    bool  domainVisible = false;

    std::vector<SurfaceFace> faces;
    std::vector<float>       faceVals;
    float                    faceDl = 0.0f;
    float                    faceNorm = 0.0f;   // robust (percentile) scale

    VoxelGridSpec      planeGrid;
    int                planeAxis = -1, planeIdx = 0, planeN1 = 0, planeN2 = 0;
    std::vector<float> planeVals;

    // far-field pattern
    int                patName = 0;     // reserved
    int                patNTheta = 0, patNPhi = 0;
    std::vector<float> patU;
    float              patUMax = 0.0f;
    Vec3               patCenter;
    float              patScale = 1.0f;

    // frame capture
    bool                       capturePending = false;
    std::vector<unsigned char> captureBuf;
    int                        captureW = 0, captureH = 0;

    // plane-wave source glyph
    bool pwShow = false;
    Aabb pwDomain;
    int  pwAxis = 0, pwPol = 2;

    // FEM mesh wireframe
    std::vector<Vec3> meshSegs;

    // MoM wire currents
    std::vector<Vec3> wirePts;
    std::vector<float> wireMag;
    float wireNorm = 0.0f;

    // MoM surface currents
    std::vector<Vec3> triCurV;
    std::vector<int>  triCurIdx;
    std::vector<float> triCurMag;
    float triCurNorm = 0.0f;
};

#endif
