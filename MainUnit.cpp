//---------------------------------------------------------------------------
// MainUnit.cpp - RF Simulator main window (UI built in code)
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "MainUnit.h"
#include "ChartForm.h"
#include <algorithm>
#include <cmath>
#include <complex>

#pragma package(smart_init)
#pragma resource "*.dfm"
TMainForm *MainForm;

static const double C0 = 299792458.0;

static double ParseD(const String &s, double def)
{
    String t = s.Trim();
    t = StringReplace(t, ",", ".", TReplaceFlags() << rfReplaceAll);
    wchar_t *end = nullptr;
    double v = wcstod(t.c_str(), &end);
    if (end == t.c_str())
        return def;
    return v;
}

//---------------------------------------------------------------------------
static String FmtD(double v)
{
    return String().sprintf(L"%g", v);
}

//---------------------------------------------------------------------------
// physical RAM available to allocations, in bytes (0 if the query fails)
static unsigned long long AvailablePhysBytes()
{
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? ms.ullAvailPhys : 0ULL;
}

//---------------------------------------------------------------------------
__fastcall TMainForm::TMainForm(TComponent* Owner)
    : TForm(Owner)
{
    // 3D viewport: custom OpenGL control, hosted in the DFM placeholder panel
    glView = new TGLView(this);
    glView->Parent = pnlView;
    glView->Align = alClient;

    // component palette entries come from the AntennaKind enum
    cbAddKind->Items->BeginUpdate();
    cbAddKind->Items->Clear();
    for (int k = 0; k < (int)AntennaKind::ImportedStl; ++k)
        cbAddKind->Items->Add(AntennaKindName((AntennaKind)k));
    cbAddKind->Items->EndUpdate();
    cbAddKind->ItemIndex = (int)AntennaKind::Yagi;

    // sanity check of the SCN scatter matrix (must be unitary)
    float err = TlmSolver::ScatterUnitarityError();
    statusBar->Panels->Items[0]->Text =
        String().sprintf(L"Ready (scatter unitarity err %.2g). Add a component "
                         L"or import an STL model to begin.", err);
}

//---------------------------------------------------------------------------
__fastcall TMainForm::~TMainForm()
{
    stopSimulation(true);
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnFormClose(TObject *, TCloseAction &)
{
    uiTimer->Enabled = false;
    if (recordingGif)
        stopGifRecording();
    stopSimulation(true);
}

//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// scene management
//---------------------------------------------------------------------------
double TMainForm::simValue(const String &key, double def)
{
    return ParseD(vleSim->Values[key], def);
}

//---------------------------------------------------------------------------
void TMainForm::setSimValue(const String &key, const String &v)
{
    vleSim->Values[key] = v;
}

//---------------------------------------------------------------------------
int TMainForm::selectedIndex()
{
    int i = lbObjects->ItemIndex;
    return (i >= 0 && i < (int)objects.size()) ? i : -1;
}

//---------------------------------------------------------------------------
void TMainForm::addAntenna(AntennaKind kind)
{
    double f0 = simValue(L"Frequency (MHz)", 1000.0) * 1e6;
    if (f0 <= 0)
        f0 = 1e9;
    invalidateResults();
    ObjEntry e;
    e.obj = CreateAntenna(kind, f0, {});
    e.obj.name += " " + std::to_string(objects.size() + 1);
    objects.push_back(std::move(e));
    refreshObjectList((int)objects.size() - 1);
    updateSceneView();
    if (objects.size() == 1)
        glView->zoomExtents();
}

//---------------------------------------------------------------------------
void TMainForm::refreshObjectList(int select)
{
    lbObjects->Items->BeginUpdate();
    lbObjects->Items->Clear();
    for (const auto &e : objects)
        lbObjects->Items->Add(String(e.obj.name.c_str()));
    lbObjects->Items->EndUpdate();
    if (select >= 0 && select < (int)objects.size())
        lbObjects->ItemIndex = select;
    refreshPropEditor();
}

//---------------------------------------------------------------------------
void TMainForm::refreshPropEditor()
{
    vleProps->Strings->BeginUpdate();
    vleProps->Strings->Clear();
    int i = selectedIndex();
    if (i >= 0)
    {
        const SceneObject &o = objects[i].obj;
        vleProps->InsertRow(L"Pos X (m)", FmtD(o.position.x), true);
        vleProps->InsertRow(L"Pos Y (m)", FmtD(o.position.y), true);
        vleProps->InsertRow(L"Pos Z (m)", FmtD(o.position.z), true);
        vleProps->InsertRow(L"Rot X (deg)", FmtD(o.rotDeg.x), true);
        vleProps->InsertRow(L"Rot Y (deg)", FmtD(o.rotDeg.y), true);
        vleProps->InsertRow(L"Rot Z (deg)", FmtD(o.rotDeg.z), true);
        if (!objects[i].isStl)
            vleProps->InsertRow(L"Design freq (MHz)", FmtD(o.designFreqHz / 1e6), true);
        for (const auto &p : o.params)
            vleProps->InsertRow(String(p.first.c_str()), FmtD(p.second), true);
    }
    vleProps->Strings->EndUpdate();
}

//---------------------------------------------------------------------------
void TMainForm::applyProperties()
{
    int i = selectedIndex();
    if (i < 0)
        return;
    invalidateResults();
    ObjEntry &e = objects[i];
    Vec3 pos((float)ParseD(vleProps->Values[L"Pos X (m)"], e.obj.position.x),
             (float)ParseD(vleProps->Values[L"Pos Y (m)"], e.obj.position.y),
             (float)ParseD(vleProps->Values[L"Pos Z (m)"], e.obj.position.z));
    Vec3 rot((float)ParseD(vleProps->Values[L"Rot X (deg)"], e.obj.rotDeg.x),
             (float)ParseD(vleProps->Values[L"Rot Y (deg)"], e.obj.rotDeg.y),
             (float)ParseD(vleProps->Values[L"Rot Z (deg)"], e.obj.rotDeg.z));

    // collect parameter rows back
    std::vector<std::pair<std::string, double>> params;
    for (int r = 0; r < vleProps->Strings->Count; ++r)
    {
        String key = vleProps->Strings->Names[r];
        if (key.Pos(L"Pos ") == 1 || key.Pos(L"Rot ") == 1 ||
            key.Pos(L"Design freq") == 1)
            continue;
        AnsiString ak(key);
        params.push_back({ ak.c_str(),
                           ParseD(vleProps->Strings->ValueFromIndex[r], 0.0) });
    }

    if (e.isStl)
    {
        double sc = GetParam(params, "Scale (m/unit)", 1.0);
        e.obj.mesh = e.stlBase;
        for (auto &v : e.obj.mesh.verts)
            v *= (float)sc;
        e.obj.mesh.computeNormals();
        e.obj.params.clear();
        e.obj.params.push_back({ "Scale (m/unit)", sc });
    }
    else
    {
        double f0 = ParseD(vleProps->Values[L"Design freq (MHz)"],
                           e.obj.designFreqHz / 1e6) * 1e6;
        if (f0 <= 0)
            f0 = 1e9;
        std::string keepName = e.obj.name;
        e.obj = CreateAntenna((AntennaKind)e.obj.kind, f0, params);
        e.obj.name = keepName;
    }
    // bake the rotation into the (freshly rebuilt) geometry, then translate
    e.obj.rotDeg = rot;
    RotateSceneObject(e.obj, rot);
    e.obj.position = pos;
    refreshPropEditor();
    updateSceneView();
}

//---------------------------------------------------------------------------
// Any scene change makes previous solver results stale: stop the run and
// clear current/field overlays so deleted geometry doesn't leave ghosts.
void TMainForm::invalidateResults()
{
    stopSimulation(true);
    solver.reset();
    haveGrid  = false;
    dftLoaded = false;
    femSolver.reset();
    usingFem = false;
    momSolver.reset();
    usingMom = false;
    momSurf.reset();
    usingMomSurf = false;
    usingSweep = false;
    meshPreviewShown = false;
    glView->clearSurfaceData();
    glView->clearPlaneData();
    glView->clearPattern();
    glView->clearMeshEdges();
    glView->clearWireCurrents();
    glView->clearTriCurrents();
    glView->setDomain(Aabb(), false);
    resetPlayback(false);
    updatePwMarker();
    btnRun->Enabled  = true;
    btnStop->Enabled = false;
    statusBar->Panels->Items[1]->Text = L"";
    statusBar->Panels->Items[2]->Text = L"";
    statusBar->Panels->Items[3]->Text = L"";
}

//---------------------------------------------------------------------------
void TMainForm::updateSceneView()
{
    std::vector<SceneObject> tmp;
    tmp.reserve(objects.size());
    for (const auto &e : objects)
        tmp.push_back(e.obj);
    glView->setScene(tmp, selectedIndex());
    updatePwMarker();
}

//---------------------------------------------------------------------------
// plane-wave source marker (horn glyph outside the domain box)
//---------------------------------------------------------------------------

// Voxelize the scene materials (dielectrics first so PEC wins overlaps).
// Used by both startSimulation() and the mesh preview.
void TMainForm::voxelizeScene(const VoxelGridSpec &g,
                              std::vector<uint8_t> &mat,
                              std::vector<MatProps> &table)
{
    mat.assign((size_t)g.nx * g.ny * g.nz, MAT_AIR);
    table.clear();
    for (const auto &e : objects)
    {
        if (!e.obj.dielectric)
            continue;
        if (table.size() >= 250)
            break;
        uint8_t id = (uint8_t)(MAT_DIEL0 + table.size());
        table.push_back({ e.obj.epsr, e.obj.sigma });
        if (!e.obj.mesh.verts.empty())
            VoxelizeMeshSolid(e.obj.mesh, e.obj.position, g, mat, id);
        for (const auto &w : e.obj.wires)
            VoxelizeWire(w, e.obj.position, g, mat, id);
    }
    for (const auto &e : objects)
    {
        if (e.obj.dielectric)
            continue;
        if (!e.obj.mesh.verts.empty())
            VoxelizeMesh(e.obj.mesh, e.obj.position, g, mat, MAT_PEC);
        for (const auto &w : e.obj.wires)
            VoxelizeWire(w, e.obj.position, g, mat, MAT_PEC);
    }
}

//---------------------------------------------------------------------------
// Derive the simulation grid from current settings + scene bounds, without
// voxelizing. Keep the math in sync with startSimulation().
bool TMainForm::computeGridPreview(VoxelGridSpec &g)
{
    if (objects.empty())
        return false;
    double f0 = simValue(L"Frequency (MHz)", 1000.0) * 1e6;
    if (f0 <= 0)
        return false;
    double lam = C0 / f0;
    int cpl = std::max(8, std::min(20000, (int)simValue(L"Cells per lambda", 20)));
    float dl = (float)(lam / cpl);
    int pad = std::max(4, std::min(80, (int)simValue(L"Padding (cells)", 12)));

    Aabb sb;
    for (const auto &e : objects)
        sb.grow(e.obj.worldBounds());
    if (!sb.valid())
        return false;

    g.dl = dl;
    Vec3 sz = sb.size();
    g.nx = (int)std::ceil(sz.x / dl) + 1 + 2 * pad;
    g.ny = (int)std::ceil(sz.y / dl) + 1 + 2 * pad;
    g.nz = (int)std::ceil(sz.z / dl) + 1 + 2 * pad;
    Vec3 c = sb.center();
    g.origin = Vec3(c.x - g.nx * dl * 0.5f,
                    c.y - g.ny * dl * 0.5f,
                    c.z - g.nz * dl * 0.5f);
    return true;
}

//---------------------------------------------------------------------------
void TMainForm::updatePwMarker()
{
    bool pw = (cbExcitation->ItemIndex == 1);
    VoxelGridSpec g;
    if (pw && computeGridPreview(g))
    {
        int pa = std::max(0, std::min(2, (int)simValue(L"PW prop axis (0..2)", 0)));
        int pol = std::max(0, std::min(2, (int)simValue(L"PW pol axis (0..2)", 2)));
        if (pol == pa)
            pol = (pa + 2) % 3;
        Aabb dom;
        dom.grow(g.origin);
        dom.grow(g.origin + Vec3(g.nx * g.dl, g.ny * g.dl, g.nz * g.dl));
        glView->setPlaneWaveMarker(dom, pa, pol, true);
        glView->setDomain(dom, true);
        static const wchar_t *axn[3] = { L"X", L"Y", L"Z" };
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"Plane wave: enters at the -%s face (horn marker), travels +%s, "
            L"E-field along %s (red arrows).", axn[pa], axn[pa], axn[pol]);
    }
    else
    {
        glView->setPlaneWaveMarker(Aabb(), 0, 2, false);
        if (!haveGrid)
            glView->setDomain(Aabb(), false);
    }
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnExcitationChanged(TObject *)
{
    updatePwMarker();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnSimSettingEdited(TObject *, System::LongInt,
                                              System::LongInt,
                                              const System::UnicodeString)
{
    if (cbExcitation->ItemIndex == 1)
        updatePwMarker();
}

//---------------------------------------------------------------------------
// simulation
//---------------------------------------------------------------------------
void TMainForm::startSimulation()
{
    if (objects.empty())
    {
        MessageDlg(L"Add a component or import an STL model first.",
                   mtInformation, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    stopSimulation(true);

    double f0 = simValue(L"Frequency (MHz)", 1000.0) * 1e6;
    if (f0 <= 0)
    {
        MessageDlg(L"Invalid frequency.", mtError, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    double lam = C0 / f0;
    int cpl = (int)simValue(L"Cells per lambda", 20);
    cpl = std::max(8, std::min(20000, cpl));
    float dl = (float)(lam / cpl);
    int pad = (int)simValue(L"Padding (cells)", 12);
    pad = std::max(4, std::min(80, pad));

    Aabb sb;
    for (const auto &e : objects)
        sb.grow(e.obj.worldBounds());
    if (!sb.valid())
        return;

    VoxelGridSpec g;
    g.dl = dl;
    Vec3 sz = sb.size();
    g.nx = (int)std::ceil(sz.x / dl) + 1 + 2 * pad;
    g.ny = (int)std::ceil(sz.y / dl) + 1 + 2 * pad;
    g.nz = (int)std::ceil(sz.z / dl) + 1 + 2 * pad;
    Vec3 c = sb.center();
    g.origin = Vec3(c.x - g.nx * dl * 0.5f,
                    c.y - g.ny * dl * 0.5f,
                    c.z - g.nz * dl * 0.5f);

    long long cells = (long long)g.nx * g.ny * g.nz;
    // per-cell working memory: TLM keeps 12 link pulses/cell, FDTD keeps 6
    // staggered fields plus Mur snapshot planes. The limit is the machine's
    // free RAM, not a fixed cell count - big TLM grids just take longer.
    bool anyDiel = false;
    for (const auto &e : objects)
        if (e.obj.dielectric) { anyDiel = true; break; }
    double bytesPerCell = (cbSolver->ItemIndex == 1)
        ? (anyDiel ? 120.0 : 96.0)            // FDTD (fields + Mur planes)
        : (anyDiel ? 64.0  : 52.0);           // TLM (12 pulses + mat)
    double needMb = cells * bytesPerCell / (1024.0 * 1024.0);
    double memMb  = needMb;

    bool timeDomain = (cbSolver->ItemIndex == 0 || cbSolver->ItemIndex == 1);
    if (timeDomain)
    {
        unsigned long long avail = AvailablePhysBytes();
        double availMb = avail / (1024.0 * 1024.0);
        if (avail > 0 && needMb > 0.85 * availMb)
        {
            MessageDlg(String().sprintf(
                L"Grid needs ~%.0f MB but only %.0f MB RAM is free.\n"
                L"%d x %d x %d = %.0f M cells.\n\n"
                L"Close other applications, reduce cells/lambda or padding, "
                L"or add RAM.",
                needMb, availMb, g.nx, g.ny, g.nz, cells / 1e6),
                mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        if (needMb > 1500.0 &&
            MessageDlg(String().sprintf(
                L"Large grid: %d x %d x %d = %.0f M cells, ~%.1f GB RAM.\n"
                L"This will run but may take a while. Continue?",
                g.nx, g.ny, g.nz, cells / 1e6, needMb / 1024.0),
                mtConfirmation, TMsgDlgButtons() << mbYes << mbNo, 0) != mrYes)
            return;
    }

    // ---- voxelize geometry ----
    std::vector<uint8_t> mat;
    std::vector<MatProps> matTable;
    voxelizeScene(g, mat, matTable);

    // ---- excitation ----
    struct FeedSet { std::vector<size_t> cells; int pol; };
    std::vector<FeedSet> feedSets;
    bool planeWave = (cbExcitation->ItemIndex == 1);
    if (!planeWave)
    {
        for (const auto &e : objects)
        {
            if (!e.obj.feed.enabled)
                continue;
            Vec3 a = e.obj.feed.a + e.obj.position;
            Vec3 b = e.obj.feed.b + e.obj.position;
            Vec3 d = b - a;
            // carve the gap and collect source cells
            Vec3 a2 = a + d * 0.15f, b2 = b - d * 0.15f;
            float len = (b2 - a2).length();
            int steps = std::max(1, (int)std::ceil(len / (dl * 0.25f)));
            FeedSet fs;
            for (int s = 0; s <= steps; ++s)
            {
                Vec3 pp = a2 + (b2 - a2) * ((float)s / steps);
                int ci, cj, ck;
                g.cellOf(pp, ci, cj, ck);
                if (!g.inGrid(ci, cj, ck))
                    continue;
                size_t cell = g.cellIndex(ci, cj, ck);
                mat[cell] = MAT_AIR;
                fs.cells.push_back(cell);
            }
            std::sort(fs.cells.begin(), fs.cells.end());
            fs.cells.erase(std::unique(fs.cells.begin(), fs.cells.end()),
                           fs.cells.end());
            // dominant axis of the gap = excited E polarization
            fs.pol = 0;
            if (std::fabs(d.y) > std::fabs(d[fs.pol])) fs.pol = 1;
            if (std::fabs(d.z) > std::fabs(d[fs.pol])) fs.pol = 2;
            if (!fs.cells.empty())
                feedSets.push_back(std::move(fs));
        }
        if (feedSets.empty())
        {
            MessageDlg(L"No feed found. Add a fed component (dipole, Yagi, "
                       L"horn, ...) or switch the excitation to plane wave.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
    }

    // ---- MoM (thin-wire EFIE) path ----
    if (cbSolver->ItemIndex == 3)
    {
        if (planeWave)
        {
            MessageDlg(L"The wire MoM solver uses a wire-port (delta-gap) "
                       L"feed. Switch the excitation to a wire port, or use "
                       L"TLM/FDTD for plane waves.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        std::vector<std::vector<Vec3>> polys;
        bool momFeed = false;
        Vec3 fa, fb;
        for (const auto &e : objects)
        {
            for (const auto &w : e.obj.wires)
            {
                std::vector<Vec3> pl;
                for (const auto &p : w.pts)
                    pl.push_back(p + e.obj.position);
                if (pl.size() >= 2)
                    polys.push_back(pl);
            }
            if (!momFeed && e.obj.feed.enabled)
            {
                momFeed = true;
                fa = e.obj.feed.a + e.obj.position;
                fb = e.obj.feed.b + e.obj.position;
            }
        }
        if (polys.empty() || !momFeed)
        {
            MessageDlg(L"The wire MoM solver needs a fed wire antenna "
                       L"(dipole, Yagi, LPDA, helix, monopole). Surface "
                       L"objects will be supported by the surface MoM.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        usingMom = true;
        usingFem = false;
        momSolver.reset(new MomSolver());
        int spl = std::max(20, cpl);
        momSolver->setupWire(polys, momFeed, fa, fb, 0.0f, (float)f0, spl,
                             chkGpu->Checked);
        haveGrid  = false;
        dftLoaded = false;
        glView->clearPattern();
        glView->clearSurfaceData();
        glView->clearPlaneData();
        glView->clearMeshEdges();
        glView->clearWireCurrents();
        glView->setDomain(Aabb(), false);
        resetPlayback(false);
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"MoM: assembling wire mesh and solving at %.4g GHz...", f0 / 1e9);
        MomSolver *ms = momSolver.get();
        solverThread = std::thread([ms] { ms->run(); });
        threadJoined = false;
        btnRun->Enabled  = false;
        btnStop->Enabled = true;
        return;
    }
    usingMom = false;
    momSolver.reset();

    // ---- surface MoM (RWG EFIE, plane-wave scattering) path ----
    if (cbSolver->ItemIndex == 4)
    {
        if (!planeWave)
        {
            MessageDlg(L"The surface MoM solver computes plane-wave "
                       L"scattering. Switch the excitation to Plane wave.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        TriMesh combined;
        for (const auto &e : objects)
        {
            if (e.obj.dielectric || e.obj.mesh.verts.empty())
                continue;
            TriMesh m = e.obj.mesh;
            m.transform(e.obj.position);
            combined.append(m);
        }
        if (combined.triCount() < 2)
        {
            MessageDlg(L"The surface MoM solver needs a PEC surface object "
                       L"(plate, box, sphere, horn, or imported STL).",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        int pa = std::max(0, std::min(2, (int)simValue(L"PW prop axis (0..2)", 0)));
        int pol = std::max(0, std::min(2, (int)simValue(L"PW pol axis (0..2)", 2)));
        if (pol == pa) pol = (pa + 2) % 3;
        momSurf.reset(new MomSurface());
        Screen->Cursor = crHourGlass;
        momSurf->setup(combined, pa, pol, (float)f0, chkGpu->Checked);
        Screen->Cursor = crDefault;
        int momN = momSurf->numUnknowns();
        if (momN == 0)
        {
            momSurf.reset();
            MessageDlg(L"This mesh produced no RWG basis functions - it is "
                       L"likely not watertight/manifold (gaps or loose "
                       L"triangles). MoM needs a closed surface. Try a "
                       L"cleaner STL or repair the mesh.",
                       mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        if ((double)momN * momN * 16.0 > 4.0e9)
        {
            momSurf.reset();
            MessageDlg(String().sprintf(
                L"Mesh is too detailed for MoM even after decimation "
                L"(%d unknowns -> %.1f GB matrix). Use a coarser STL.",
                momN, momN * (double)momN * 16.0 / 1e9),
                mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        usingMomSurf = true;
        usingFem = false; usingMom = false;
        haveGrid = false; dftLoaded = false;
        glView->clearPattern();
        glView->clearSurfaceData();
        glView->clearPlaneData();
        glView->clearMeshEdges();
        glView->clearWireCurrents();
        glView->clearTriCurrents();
        glView->setDomain(Aabb(), false);
        resetPlayback(false);
        String decNote = momSurf->reducedFromTris() > 0
            ? String().sprintf(L"decimated %d -> %d tris, ",
                momSurf->reducedFromTris(), momSurf->numTris())
            : String();
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"Surface MoM: %s%d RWG unknowns, solving at %.4g GHz...",
            decNote.c_str(), momN, f0 / 1e9);
        MomSurface *ms = momSurf.get();
        solverThread = std::thread([ms] { ms->run(); });
        threadJoined = false;
        btnRun->Enabled  = false;
        btnStop->Enabled = true;
        return;
    }
    usingMomSurf = false;
    momSurf.reset();

    // ---- FEM (frequency domain) path ----
    if (cbSolver->ItemIndex == 2)
    {
        if (planeWave)
        {
            MessageDlg(L"The FEM solver currently supports wire-port "
                       L"excitation only. Switch the excitation to a wire "
                       L"port, or use the TLM solver for plane waves.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        if (cells * 7 > 1500000LL)
        {
            MessageDlg(String().sprintf(
                L"FEM mesh would have ~%.1f M edge unknowns. Reduce cells "
                L"per lambda (FEM is accurate at 10-15) or padding.",
                cells * 7 / 1e6), mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        usingFem = true;
        femSolver.reset(new FemSolver());
        femSolver->setup(g, std::move(mat), std::move(matTable), (float)f0,
                         feedSets[0].cells, feedSets[0].pol, chkGpu->Checked);
        lastGrid = g;
        haveGrid = true;
        dftLoaded = false;
        glView->clearPattern();
        glView->clearSurfaceData();
        glView->clearPlaneData();
        glView->clearMeshEdges();
        resetPlayback(false);
        Aabb dom;
        dom.grow(g.origin);
        dom.grow(g.origin + Vec3(g.nx * dl, g.ny * dl, g.nz * dl));
        glView->setDomain(dom, true);
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"FEM: meshing %d x %d x %d cells (~%.2f M tets) and solving at "
            L"%.4g GHz...", g.nx, g.ny, g.nz, cells * 6 / 1e6, f0 / 1e9);
        FemSolver *fs = femSolver.get();
        solverThread = std::thread([fs] { fs->run(); });
        threadJoined = false;
        btnRun->Enabled  = false;
        btnStop->Enabled = true;
        return;
    }
    usingFem = false;
    femSolver.reset();

    // ---- time-domain solver config (TLM or FDTD) ----
    const bool useFdtd = (cbSolver->ItemIndex == 1);
    TlmConfig cfg;
    cfg.f0 = (float)f0;
    int wf = cbWaveform->ItemIndex;
    cfg.waveform = (wf == 1) ? WaveformType::GaussianSine
                 : (wf == 2) ? WaveformType::GaussianPulse
                             : WaveformType::CwRamped;
    cfg.boundaryRho = (float)std::max(-1.0, std::min(1.0,
                          simValue(L"Boundary rho", 0.0)));
    int spp = 2 * cpl;   // timesteps per period (dt = dl/2c)
    int settle = (int)simValue(L"Settle (0:auto)", 0);
    if (settle <= 0)
        settle = 2 * (g.nx + g.ny + g.nz) + 3 * spp;
    int total = (int)simValue(L"Timesteps (0:auto)", 0);
    if (total <= 0)
    {
        // pulsed runs need ringdown time for clean port spectra
        int tail = (cfg.waveform == WaveformType::CwRamped) ? 12 : 30;
        total = settle + tail * spp;
    }
    if (total <= settle)
        total = settle + 4 * spp;
    cfg.settleSteps = settle;
    cfg.totalSteps  = total;
    cfg.huyOffset   = std::max(2, pad / 2);
    cfg.useGpu      = chkGpu->Checked;

    if (useFdtd)
        solver.reset(new FdtdSolver());
    else
        solver.reset(new TlmSolver());
    solver->setup(g, std::move(mat), std::move(matTable), cfg);

    if (planeWave)
    {
        int pa = (int)simValue(L"PW prop axis (0..2)", 0);
        pa = std::max(0, std::min(2, pa));
        int pol = (int)simValue(L"PW pol axis (0..2)", 2);
        pol = std::max(0, std::min(2, pol));
        if (pol == pa)
            pol = (pa + 2) % 3;
        solver->addPlaneWave(pa, std::max(2, pad / 2), pol, 1.0f);
    }
    else
    {
        for (const auto &fs : feedSets)
            solver->addPort(fs.cells, fs.pol, 1.0f);
    }

    // |E| cut plane through the scene center
    if (chkPlane->Checked)
    {
        int axis = (int)simValue(L"Cut plane axis", 1);
        axis = std::max(0, std::min(2, axis));
        int ci, cj, ck;
        g.cellOf(sb.center(), ci, cj, ck);
        int idx = (axis == 0) ? ci : (axis == 1) ? cj : ck;
        const int nn[3] = { g.nx, g.ny, g.nz };
        idx = std::max(0, std::min(nn[axis] - 1, idx));
        solver->setFieldPlane(axis, idx);
    }
    else
        solver->setFieldPlane(-1, 0);

    // ---- playback frame recording (memory-budgeted) ----
    {
        size_t nfaces = solver->faces().size();
        size_t planeMax = chkPlane->Checked
            ? (size_t)std::max(g.nx * g.ny, std::max(g.ny * g.nz, g.nx * g.nz))
            : 0;
        size_t frameBytes = nfaces * 4 + planeMax * 4 + 256;
        long long budget = 300LL << 20;             // ~300 MB
        int maxFrames = (int)std::min<long long>(500, budget / (long long)frameBytes);
        int recEvery = 0;
        if (maxFrames >= 10)
        {
            recEvery = (total + maxFrames - 1) / maxFrames;
            recEvery = ((recEvery + 1) / 2) * 2;    // even: cut plane refresh
            if (recEvery < 2)
                recEvery = 2;
        }
        solver->setRecordEvery(recEvery);
    }
    resetPlayback(false);

    lastGrid = g;
    haveGrid = true;
    dftLoaded = false;
    glView->clearPattern();
    Aabb dom;
    dom.grow(g.origin);
    dom.grow(g.origin + Vec3(g.nx * dl, g.ny * dl, g.nz * dl));
    glView->setDomain(dom, true);

    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"Running %s: %d x %d x %d cells (%.1f M, %.0f MB), dt=%.3g ps, "
        L"%d steps", solver->solverName(), g.nx, g.ny, g.nz, cells / 1e6,
        memMb, (double)solver->timestep() * 1e12, total);

    IFieldSolver *s = solver.get();
    solverThread = std::thread([s] { s->run(); });
    threadJoined = false;
    btnRun->Enabled  = false;
    btnStop->Enabled = true;
}

//---------------------------------------------------------------------------
void TMainForm::stopSimulation(bool wait)
{
    if (solver)
        solver->requestStop();
    if (femSolver)
        femSolver->requestStop();
    if (momSolver)
        momSolver->requestStop();
    if (momSurf)
        momSurf->requestStop();
    if (wait)
        finishThread();
}

//---------------------------------------------------------------------------
void TMainForm::finishThread()
{
    if (!threadJoined && solverThread.joinable())
    {
        solverThread.join();
        threadJoined = true;
    }
}

//---------------------------------------------------------------------------
// visualization refresh
//---------------------------------------------------------------------------
void TMainForm::updateVisualization()
{
    if (usingFem || usingMom || usingMomSurf)
        return;             // FEM/MoM display is set once on finish
    if (!solver || !haveGrid)
        return;
    if (playbackMode && solver->isFinished())
        return;             // the scrubber owns the display
    int mode = cbCurrents->ItemIndex;       // 0 off, 1 live, 2 DFT
    if (mode == 2 && solver->isFinished())
    {
        std::vector<float> v;
        if (solver->getJsDft(v))
            glView->setSurfaceData(solver->faces(), v, lastGrid.dl);
    }
    else if (mode != 0)
    {
        std::vector<float> v;
        solver->getJsInstant(v);
        glView->setSurfaceData(solver->faces(), v, lastGrid.dl);
    }
    else
        glView->clearSurfaceData();

    if (chkPlane->Checked)
    {
        std::vector<float> pv;
        int axis, idx, n1, n2;
        if (solver->getFieldPlane(pv, axis, idx, n1, n2))
            glView->setPlaneData(lastGrid, axis, idx, n1, n2, pv);
    }
    else
        glView->clearPlaneData();
}

//---------------------------------------------------------------------------
void TMainForm::updateMeshView()
{
    if (!chkMesh->Checked)
    {
        meshPreviewShown = false;
        glView->clearMeshEdges();
        return;
    }
    if (usingFem && femSolver && femSolver->isFinished())
    {
        std::vector<Vec3> segs;
        femSolver->getMeshViz(segs);
        glView->setMeshEdges(segs);
    }
    else if (!meshPreviewShown)
        glView->clearMeshEdges();
    // else: keep the standalone mesh preview on screen
}

//---------------------------------------------------------------------------
// "Generate mesh": build and display the discretization without solving,
// so the problem size can be judged before committing.
//   TLM/FDTD: voxel-block outline of all conductor/dielectric cells
//   FEM:      tetrahedral wireframe + unknown count
//---------------------------------------------------------------------------
void __fastcall TMainForm::OnMeshClick(TObject *)
{
    // ---- surface MoM: show the decimated RWG triangulation ----
    if (cbSolver->ItemIndex == 4)
    {
        TriMesh combined;
        for (const auto &e : objects)
        {
            if (e.obj.dielectric || e.obj.mesh.verts.empty())
                continue;
            TriMesh m = e.obj.mesh;
            m.transform(e.obj.position);
            combined.append(m);
        }
        if (combined.triCount() < 2)
        {
            MessageDlg(L"Surface MoM needs a PEC surface object (plate, box, "
                       L"sphere, horn, or imported STL).",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        double f0 = simValue(L"Frequency (MHz)", 1000.0) * 1e6;
        if (f0 <= 0) f0 = 1e9;
        Screen->Cursor = crHourGlass;
        MomSurface tmp;
        tmp.setup(combined, 0, 0, (float)f0, false);   // decimate + build RWG
        std::vector<Vec3> v; std::vector<int> idx; std::vector<float> mag;
        tmp.getTriCurrents(v, idx, mag);
        // wireframe: 3 edges per triangle
        std::vector<Vec3> segs;
        segs.reserve(idx.size() * 2);
        for (size_t t = 0; t + 2 < idx.size(); t += 3)
            for (int e = 0; e < 3; ++e)
            {
                segs.push_back(v[idx[t + e]]);
                segs.push_back(v[idx[t + (e + 1) % 3]]);
            }
        Screen->Cursor = crDefault;
        glView->setMeshEdges(segs);
        chkMesh->Checked = true;
        meshPreviewShown = true;
        glView->setDomain(Aabb(), false);
        String decNote = tmp.reducedFromTris() > 0
            ? String().sprintf(L"decimated %d -> %d tris, ",
                tmp.reducedFromTris(), tmp.numTris())
            : String().sprintf(L"%d tris, ", tmp.numTris());
        double mb = tmp.numUnknowns() * (double)tmp.numUnknowns() * 16.0 / 1e6;
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"MoM surface mesh: %s%d RWG unknowns (dense matrix ~%.0f MB). "
            L"Wireframe shows the MoM triangulation.",
            decNote.c_str(), tmp.numUnknowns(), mb);
        return;
    }

    VoxelGridSpec g;
    if (!computeGridPreview(g))
    {
        MessageDlg(L"Add a component or import an STL model first.",
                   mtInformation, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    long long cells = (long long)g.nx * g.ny * g.nz;
    const bool femSel = (cbSolver->ItemIndex == 2);
    if (femSel && cells * 7 > 4000000LL)
    {
        MessageDlg(String().sprintf(
            L"This mesh would have ~%.1f M edges - too large to preview or "
            L"solve. Reduce cells per lambda (FEM is accurate at 10-15) or "
            L"padding.", cells * 7 / 1e6),
            mtError, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    if (!femSel && cells > 800000000LL)   // preview needs only ~1 byte/cell
    {
        MessageDlg(L"Grid is too large to preview. Reduce cells per lambda "
                   L"or padding.", mtError, TMsgDlgButtons() << mbOK, 0);
        return;
    }

    Screen->Cursor = crHourGlass;
    std::vector<uint8_t> mat;
    std::vector<MatProps> table;
    voxelizeScene(g, mat, table);

    if (femSel)
    {
        FemMesh mesh;
        bool ok = mesh.build(g, mat);
        Screen->Cursor = crDefault;
        if (!ok)
        {
            MessageDlg(L"Mesh generation failed (empty or degenerate grid).",
                       mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        size_t pecEdges = 0;
        for (uint8_t e : mesh.edgePec)
            pecEdges += e;
        size_t unknowns = mesh.edges.size() - pecEdges;
        double solveMb = unknowns * 33.0 * 16.0 / 1e6;   // CSR estimate
        std::vector<Vec3> segs;
        mesh.vizEdges(mat, segs, 300000);
        glView->setMeshEdges(segs);
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"FEM mesh preview: %u nodes, %u tets, %u edges -> %u unknowns "
            L"(matrix ~%.0f MB). Wireframe shows conductor/dielectric cells.",
            (unsigned)mesh.nodes.size(), (unsigned)mesh.tets.size(),
            (unsigned)mesh.edges.size(), (unsigned)unknowns, solveMb);
    }
    else
    {
        // voxel-block outline for the time-domain solvers
        std::vector<Vec3> segs;
        BuildVoxelOutline(g, mat, segs, 400000);
        Screen->Cursor = crDefault;
        size_t nPec = 0, nDiel = 0;
        for (uint8_t m : mat)
        {
            if (m == MAT_PEC)
                ++nPec;
            else if (m >= MAT_DIEL0)
                ++nDiel;
        }
        glView->setMeshEdges(segs);

        const bool fdtd = (cbSolver->ItemIndex == 1);
        double bytesPerCell = fdtd ? 27.0 : 49.0;   // field arrays + masks
        int cpl = std::max(8, std::min(20000,
                  (int)simValue(L"Cells per lambda", 20)));
        int spp = 2 * cpl;
        int autoSteps = 2 * (g.nx + g.ny + g.nz) + 3 * spp + 12 * spp;
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"%s grid preview: %d x %d x %d cells (%.2f M, ~%.0f MB), "
            L"cell = %.2f mm (%d / lambda), ~%d auto steps. "
            L"%u PEC + %u dielectric blocks outlined.",
            fdtd ? L"FDTD" : L"TLM", g.nx, g.ny, g.nz, cells / 1e6,
            cells * bytesPerCell / 1e6, g.dl * 1000.0, cpl, autoSteps,
            (unsigned)nPec, (unsigned)nDiel);
    }

    chkMesh->Checked = true;
    meshPreviewShown = true;
    Aabb dom;
    dom.grow(g.origin);
    dom.grow(g.origin + Vec3(g.nx * g.dl, g.ny * g.dl, g.nz * g.dl));
    glView->setDomain(dom, true);
}

//---------------------------------------------------------------------------
void TMainForm::finishFemRun()
{
    btnRun->Enabled  = true;
    btnStop->Enabled = false;
    std::vector<float> js;
    femSolver->getJs(js);
    glView->setSurfaceData(femSolver->faces(), js, lastGrid.dl);
    if (chkPlane->Checked)
    {
        int axis = std::max(0, std::min(2, (int)simValue(L"Cut plane axis", 1)));
        const int nn[3] = { lastGrid.nx, lastGrid.ny, lastGrid.nz };
        int n1, n2;
        std::vector<float> pv;
        if (femSolver->getFieldPlane(axis, nn[axis] / 2, n1, n2, pv))
            glView->setPlaneData(lastGrid, axis, nn[axis] / 2, n1, n2, pv);
    }
    updateMeshView();

    String zs = L"no port current";
    if (femSolver->zinValid())
    {
        std::complex<double> z = femSolver->zin();
        zs = String().sprintf(L"Zin = %.1f %s j%.1f ohm", z.real(),
                              z.imag() < 0 ? L"-" : L"+",
                              std::fabs(z.imag()));
    }
    String where = femSolver->ranOnGpu()
        ? String().sprintf(L"GPU: %hs", femSolver->gpuStatus().c_str())
        : String(L"CPU");
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"FEM %s (%s):  %s @ f0.  Mesh: %u nodes, %u tets, %u edges "
        L"(%u unknowns), residual %.2g.",
        femSolver->converged() ? L"converged" : L"NOT fully converged",
        where.c_str(), zs.c_str(), (unsigned)femSolver->numNodes(),
        (unsigned)femSolver->numTets(), (unsigned)femSolver->numEdges(),
        (unsigned)femSolver->numUnknowns(), (double)femSolver->residual());
}

//---------------------------------------------------------------------------
void TMainForm::finishMomRun()
{
    btnRun->Enabled  = true;
    btnStop->Enabled = false;
    std::vector<Vec3> pts;
    std::vector<float> mag;
    momSolver->getWireCurrents(pts, mag);
    glView->setWireCurrents(pts, mag);

    String zs = L"no port";
    if (momSolver->zinValid())
    {
        std::complex<double> z = momSolver->zin();
        zs = String().sprintf(L"Zin = %.1f %s j%.1f ohm", z.real(),
                              z.imag() < 0 ? L"-" : L"+", std::fabs(z.imag()));
    }
    String where = momSolver->ranOnGpu()
        ? String().sprintf(L"GPU: %hs", momSolver->gpuStatus().c_str())
        : String(L"CPU");
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"MoM (%s):  %s @ f0.  %d segments, %d unknowns. Far-field pattern "
        L"available; wire color = |current|.",
        where.c_str(), zs.c_str(), momSolver->numSegments(),
        momSolver->numUnknowns());
}

void TMainForm::finishMomSurfRun()
{
    btnRun->Enabled  = true;
    btnStop->Enabled = false;
    std::vector<Vec3> v; std::vector<int> idx; std::vector<float> mag;
    momSurf->getTriCurrents(v, idx, mag);
    glView->setTriCurrents(v, idx, mag);
    String where = momSurf->ranOnGpu()
        ? String().sprintf(L"GPU: %hs", momSurf->gpuStatus().c_str())
        : String(L"CPU");
    String decNote = momSurf->reducedFromTris() > 0
        ? String().sprintf(L" (decimated from %d)", momSurf->reducedFromTris())
        : String();
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"Surface MoM (%s):  %d triangles%s, %d RWG unknowns. Face color = "
        L"|J|; Far-field gives the scattered pattern / RCS.",
        where.c_str(), momSurf->numTris(), decNote.c_str(),
        momSurf->numUnknowns());
}

void TMainForm::finishSweep()
{
    usingSweep = false;
    btnRun->Enabled  = true;
    btnStop->Enabled = false;
    std::vector<float> fr;
    std::vector<double> rcs;
    momSurf->getSweep(fr, rcs);
    if (fr.size() < 2)
    {
        statusBar->Panels->Items[0]->Text = L"RCS sweep produced no data.";
        return;
    }
    std::vector<double> fGhz(fr.size()), dbsm(fr.size());
    double pk = -1e30; double pkF = 0;
    for (size_t i = 0; i < fr.size(); ++i)
    {
        fGhz[i] = fr[i] / 1e9;
        dbsm[i] = 10.0 * std::log10(std::max(1e-12, rcs[i]));
        if (dbsm[i] > pk) { pk = dbsm[i]; pkF = fr[i] / 1e6; }
    }
    if (!chartForm)
        chartForm = new TChartForm(this);
    chartForm->pages.clear();
    TChartForm::Page pg;
    pg.title  = L"Monostatic RCS vs frequency";
    pg.xLabel = L"Frequency (GHz)";
    pg.yLabel = L"RCS (dBsm)";
    pg.curves.push_back({ L"RCS", fGhz, dbsm, (TColor)0x00007700 });
    chartForm->pages.push_back(pg);
    chartForm->refreshPages();
    chartForm->Show();
    chartForm->BringToFront();
    // update displayed currents to the last swept frequency
    std::vector<Vec3> v; std::vector<int> idx; std::vector<float> mag;
    momSurf->getTriCurrents(v, idx, mag);
    glView->setTriCurrents(v, idx, mag);
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"RCS sweep done: %d points. Peak %.1f dBsm near %.0f MHz.",
        (int)fr.size(), pk, pkF);
}

void __fastcall TMainForm::OnTimerTick(TObject *)
{
    if (usingSweep)
    {
        if (!momSurf)
            return;
        statusBar->Panels->Items[1]->Text = String().sprintf(
            L"RCS sweep  %d / %d", momSurf->currentStep(),
            momSurf->sweepTotal());
        statusBar->Panels->Items[2]->Text = L"";
        if (momSurf->isFinished())
        {
            finishThread();
            if (!dftLoaded) { dftLoaded = true; finishSweep(); }
        }
        return;
    }
    if (usingMomSurf)
    {
        if (!momSurf)
            return;
        statusBar->Panels->Items[1]->Text = String().sprintf(
            L"%hs  %d / %d", momSurf->phase().c_str(),
            momSurf->currentStep(), momSurf->totalSteps());
        statusBar->Panels->Items[2]->Text = String().sprintf(
            L"Residual %.3g", (double)momSurf->residual());
        if (momSurf->isFinished())
        {
            finishThread();
            if (!dftLoaded) { dftLoaded = true; finishMomSurfRun(); }
        }
        statusBar->Panels->Items[3]->Text = String().sprintf(
            L"|J|max %.3g", (double)glView->currentMax);
        return;
    }
    if (usingMom)
    {
        if (!momSolver)
            return;
        statusBar->Panels->Items[1]->Text = String().sprintf(
            L"%hs  %d / %d", momSolver->phase().c_str(),
            momSolver->currentStep(), momSolver->totalSteps());
        statusBar->Panels->Items[2]->Text = String().sprintf(
            L"Residual %.3g", (double)momSolver->residual());
        if (momSolver->isFinished())
        {
            finishThread();
            if (!dftLoaded)
            {
                dftLoaded = true;
                finishMomRun();
            }
        }
        statusBar->Panels->Items[3]->Text = String().sprintf(
            L"|I|max %.3g", (double)glView->currentMax);
        return;
    }
    if (usingFem)
    {
        if (!femSolver)
            return;
        statusBar->Panels->Items[1]->Text = String().sprintf(
            L"%hs  %d / %d", femSolver->phase().c_str(),
            femSolver->currentStep(), femSolver->totalSteps());
        statusBar->Panels->Items[2]->Text = String().sprintf(
            L"Residual %.3g", (double)femSolver->residual());
        if (femSolver->isFinished())
        {
            finishThread();
            if (!dftLoaded)
            {
                dftLoaded = true;
                finishFemRun();
            }
        }
        statusBar->Panels->Items[3]->Text = String().sprintf(
            L"|Js|max %.3g", (double)glView->currentMax);
        return;
    }
    if (!solver)
        return;
    int n = solver->currentStep(), tot = solver->totalSteps();
    statusBar->Panels->Items[1]->Text =
        String().sprintf(L"Step %d / %d", n, tot);
    statusBar->Panels->Items[2]->Text =
        String().sprintf(L"Energy %.3g", (double)solver->currentEnergy());

    if (solver->isRunning())
    {
        updateVisualization();
        if (recordingGif && n != lastGifStep)
        {
            lastGifStep = n;
            captureGifFrame();
        }
    }
    else if (solver->isFinished())
    {
        finishThread();
        if (!dftLoaded)
        {
            dftLoaded = true;
            btnRun->Enabled  = true;
            btnStop->Enabled = false;
            String how = solver->ranOnGpu()
                ? String().sprintf(L"GPU: %hs", solver->gpuStatus().c_str())
                : String(L"CPU");
            String gnote;
            if (chkGpu->Checked && !solver->ranOnGpu())
                gnote = String().sprintf(L" (GPU unavailable: %hs)",
                                         solver->gpuStatus().c_str());
            statusBar->Panels->Items[0]->Text = String().sprintf(
                L"%s finished (%s%s). 'Currents: Js @ f0' = frequency-domain "
                L"currents; Plots and Far-field are ready.",
                solver->solverName(), how.c_str(), gnote.c_str());
            updateVisualization();
            resetPlayback(true);
            if (recordingGif)
            {
                captureGifFrame();      // final frame
                stopGifRecording();
            }
        }
        else if (playing && tbPlayback->Enabled)
        {
            int next = tbPlayback->Position + 1;
            if (next > tbPlayback->Max)
                next = 0;
            tbPlayback->Position = next;   // OnChange shows the frame
        }
    }
    statusBar->Panels->Items[3]->Text =
        String().sprintf(L"|Js|max %.3g", (double)glView->currentMax);
}

//---------------------------------------------------------------------------
// event handlers
//---------------------------------------------------------------------------
void __fastcall TMainForm::OnAddClick(TObject *)
{
    int k = cbAddKind->ItemIndex;
    if (k >= 0)
        addAntenna((AntennaKind)k);
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnImportClick(TObject *)
{
    std::unique_ptr<TOpenDialog> dlg(new TOpenDialog(this));
    dlg->Filter = L"STL files (*.stl)|*.stl|All files (*.*)|*.*";
    dlg->Title = L"Import 3D model";
    if (!dlg->Execute())
        return;
    TriMesh m;
    std::string err;
    if (!LoadStl(dlg->FileName.c_str(), m, err))
    {
        MessageDlg(String().sprintf(L"STL import failed: %hs", err.c_str()),
                   mtError, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    invalidateResults();
    ObjEntry e;
    e.isStl = true;
    e.stlBase = m;
    e.obj.kind = (int)AntennaKind::ImportedStl;
    AnsiString fn(ExtractFileName(dlg->FileName));
    e.obj.name = fn.c_str();

    // STL files are commonly in millimetres; auto-scale if it looks like it
    double scale = 1.0;
    Aabb b = m.bounds();
    if (b.valid() && b.size().length() > 20.0f)
        scale = 0.001;
    e.obj.params.push_back({ "Scale (m/unit)", scale });
    e.obj.mesh = m;
    if (scale != 1.0)
    {
        for (auto &v : e.obj.mesh.verts)
            v *= (float)scale;
        e.obj.mesh.computeNormals();
    }
    objects.push_back(std::move(e));
    refreshObjectList((int)objects.size() - 1);
    updateSceneView();
    glView->zoomExtents();
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"Imported %d triangles%s", m.triCount(),
        scale != 1.0 ? L" (auto-scaled mm -> m, see 'Scale' property)" : L"");
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnDeleteClick(TObject *)
{
    int i = selectedIndex();
    if (i < 0 && objects.size() == 1)
        i = 0;                  // nothing selected but only one object: delete it
    if (i < 0)
        return;
    invalidateResults();
    objects.erase(objects.begin() + i);
    refreshObjectList(std::min(i, (int)objects.size() - 1));
    updateSceneView();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnRunClick(TObject *)
{
    startSimulation();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnStopClick(TObject *)
{
    stopSimulation(false);
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnZoomClick(TObject *)
{
    glView->zoomExtents();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnApplyClick(TObject *)
{
    applyProperties();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnObjectSelect(TObject *)
{
    refreshPropEditor();
    updateSceneView();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnViewOptionChanged(TObject *)
{
    glView->showModel = chkModel->Checked;
    glView->dbScale   = chkDb->Checked;
    glView->dbRange   = 20.0f + 10.0f * std::max(0, (int)cbDbRange->ItemIndex);
    // changing the display mode leaves playback scrubbing
    playbackMode = false;
    playing = false;
    btnPlay->Caption = L"▶ Play";
    updateMeshView();
    if (solver)
        updateVisualization();
    else
    {
        glView->clearSurfaceData();
        glView->clearPlaneData();
    }
    glView->Invalidate();
}

//---------------------------------------------------------------------------
// S11 / input impedance plots from the recorded port V(t), I(t)
//---------------------------------------------------------------------------
void TMainForm::showPlots()
{
    // surface MoM: run a monostatic RCS frequency sweep
    if (usingMomSurf)
    {
        if (!momSurf || !momSurf->isFinished())
        {
            MessageDlg(L"Run the surface MoM solve once first (it builds the "
                       L"RWG mesh the sweep reuses).",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        double f1 = simValue(L"RCS sweep f1 (MHz)", 10) * 1e6;
        double f2 = simValue(L"RCS sweep f2 (MHz)", 40) * 1e6;
        int pts = std::max(2, std::min(101,
                  (int)simValue(L"RCS sweep points", 16)));
        if (f1 <= 0 || f2 <= f1)
        {
            MessageDlg(L"Set RCS sweep f1 < f2 (MHz) in the simulation "
                       L"settings.", mtError, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        double secsEach = momSurf->ranOnGpu() ? 8.0 : 25.0;
        if (MessageDlg(String().sprintf(
            L"Run a monostatic RCS sweep: %d points from %.0f to %.0f MHz "
            L"(%d unknowns each).\nEach point is a full fill + solve "
            L"(~%.0f s) -> roughly %.0f min total. Continue?",
            pts, f1/1e6, f2/1e6, momSurf->numUnknowns(), secsEach,
            pts * secsEach / 60.0),
            mtConfirmation, TMsgDlgButtons() << mbYes << mbNo, 0) != mrYes)
            return;
        std::vector<float> freqs(pts);
        for (int i = 0; i < pts; ++i)
            freqs[i] = (float)(f1 + (f2 - f1) * i / (pts - 1));
        usingSweep = true;
        dftLoaded  = false;
        MomSurface *ms = momSurf.get();
        solverThread = std::thread([ms, freqs] { ms->runRcsSweep(freqs); });
        threadJoined = false;
        btnRun->Enabled  = false;
        btnStop->Enabled = true;
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"RCS sweep: %d points, %.0f-%.0f MHz...", pts, f1/1e6, f2/1e6);
        return;
    }
    if (usingFem || usingMom)
    {
        bool ok = usingFem ? (femSolver && femSolver->isFinished() &&
                              femSolver->zinValid())
                           : (momSolver && momSolver->isFinished() &&
                              momSolver->zinValid());
        if (ok)
        {
            std::complex<double> z = usingFem ? femSolver->zin()
                                              : momSolver->zin();
            MessageDlg(String().sprintf(
                L"%s is a single-frequency solve:\n\nZin(f0) = %.2f %s "
                L"j%.2f ohm\n\nFor S11/impedance sweeps and energy-vs-time, "
                L"use the TLM solver with a Gaussian-sine excitation.",
                usingFem ? L"FEM" : L"MoM",
                z.real(), z.imag() < 0 ? L"-" : L"+", std::fabs(z.imag())),
                mtInformation, TMsgDlgButtons() << mbOK, 0);
        }
        else
            MessageDlg(L"Run the solve first.", mtInformation,
                       TMsgDlgButtons() << mbOK, 0);
        return;
    }
    if (!solver || !solver->isFinished())
    {
        MessageDlg(L"Run a simulation first.", mtInformation,
                   TMsgDlgButtons() << mbOK, 0);
        return;
    }
    if (!chartForm)
        chartForm = new TChartForm(this);
    chartForm->pages.clear();

    // ---- energy vs time (always available) ----
    {
        std::vector<int> esteps;
        std::vector<float> evals;
        solver->getEnergyHistory(esteps, evals);
        if (evals.size() > 1)
        {
            double dtv = solver->timestep();
            std::vector<double> tNs(evals.size()), lin(evals.size()),
                                db(evals.size());
            double eMax = 0.0;
            for (float v : evals)
                eMax = std::max(eMax, (double)v);
            for (size_t i = 0; i < evals.size(); ++i)
            {
                tNs[i] = esteps[i] * dtv * 1e9;
                lin[i] = evals[i];
                db[i]  = (eMax > 0)
                    ? 10.0 * std::log10(std::max((double)evals[i],
                                                 eMax * 1e-12) / eMax)
                    : 0.0;
            }
            TChartForm::Page pg;
            pg.title  = L"Energy vs time";
            pg.xLabel = L"Time (ns)";
            pg.yLabel = L"Field energy  (a.u.)";
            pg.curves.push_back({ L"Energy", tNs, lin, (TColor)0x00007700 });
            chartForm->pages.push_back(pg);

            TChartForm::Page pgDb;
            pgDb.title  = L"Energy vs time (dB)";
            pgDb.xLabel = L"Time (ns)";
            pgDb.yLabel = L"Energy rel. peak  (dB)";
            pgDb.curves.push_back({ L"Energy dB", tNs, db, (TColor)0x00007700 });
            chartForm->pages.push_back(pgDb);
        }
    }

    // ---- S11 / impedance (wire-port runs only) ----
    const auto &ports = solver->ports();
    if (!ports.empty() && !ports[0].vRec.empty())
    {
        const TlmPort &p = ports[0];
        const int N = (int)p.vRec.size();
        const double dtv = solver->timestep();
        const double f0 = solver->cfg().f0;

        // analysis band depends on excitation bandwidth
        double fLo = 0.9 * f0, fHi = 1.1 * f0;
        switch (solver->cfg().waveform)
        {
        case WaveformType::GaussianSine:  fLo = 0.55 * f0; fHi = 1.45 * f0; break;
        case WaveformType::GaussianPulse: fLo = 0.15 * f0; fHi = 1.60 * f0; break;
        default: break;
        }
        const int nF = 241;
        std::vector<double> fGhz(nF), s11db(nF), reZ(nF), imZ(nF);
        for (int q = 0; q < nF; ++q)
        {
            double f = fLo + (fHi - fLo) * q / (nF - 1);
            std::complex<double> Vf(0, 0), If(0, 0);
            double w = 2.0 * M_PI * f * dtv;
            for (int n = 0; n < N; ++n)
            {
                std::complex<double> e(std::cos(w * n), -std::sin(w * n));
                Vf += (double)p.vRec[n] * e;
                If += (double)p.iRec[n] * e;
            }
            std::complex<double> Z = (std::abs(If) > 1e-30) ? Vf / If
                                    : std::complex<double>(0, 0);
            std::complex<double> s11 = (Z - 50.0) / (Z + 50.0);
            fGhz[q]  = f / 1e9;
            s11db[q] = 20.0 * std::log10(std::max(1e-8, std::abs(s11)));
            reZ[q]   = Z.real();
            imZ[q]   = Z.imag();
        }
        TChartForm::Page pg;
        pg.title  = L"S11 (50 ohm reference)";
        pg.yLabel = L"|S11|  (dB)";
        pg.curves.push_back({ L"S11 dB", fGhz, s11db, (TColor)0x00CC6600 });
        chartForm->pages.push_back(pg);

        TChartForm::Page pgZ;
        pgZ.title  = L"Input impedance";
        pgZ.yLabel = L"Z  (ohm)";
        pgZ.curves.push_back({ L"Re Z", fGhz, reZ, (TColor)0x000000CC });
        pgZ.curves.push_back({ L"Im Z", fGhz, imZ, (TColor)0x00CC0000 });
        chartForm->pages.push_back(pgZ);
    }

    if (chartForm->pages.empty())
    {
        MessageDlg(L"No plot data was recorded in this run.", mtInformation,
                   TMsgDlgButtons() << mbOK, 0);
        return;
    }
    chartForm->refreshPages();
    chartForm->Show();
    chartForm->BringToFront();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnPlotsClick(TObject *)
{
    showPlots();
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnFarFieldClick(TObject *)
{
    if (glView->hasPattern())
    {
        glView->clearPattern();
        return;
    }
    if (usingMomSurf)
    {
        if (!momSurf || !momSurf->isFinished())
        {
            MessageDlg(L"Run the surface MoM solve first.", mtInformation,
                       TMsgDlgButtons() << mbOK, 0);
            return;
        }
        Screen->Cursor = crHourGlass;
        FarFieldData ffs;
        bool oks = momSurf->computeFarField(ffs);
        Screen->Cursor = crDefault;
        if (!oks)
        {
            MessageDlg(L"No far-field data from the surface MoM run.",
                       mtInformation, TMsgDlgButtons() << mbOK, 0);
            return;
        }
        std::vector<Vec3> vv; std::vector<int> ii; std::vector<float> mm;
        momSurf->getTriCurrents(vv, ii, mm);
        Aabb b;
        for (const auto &p : vv) b.grow(p);
        Vec3 center = b.valid() ? b.center() : Vec3(0,0,0);
        float scale = b.valid() ? 0.6f * b.size().length() : 1.0f;
        glView->setPattern(ffs, center, scale);
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"Surface MoM scattered field:  D = %.2f dBi,  peak at theta=%.0f "
            L"deg, phi=%.0f deg  (bistatic pattern).",
            10.0 * std::log10(std::max(1e-6f, ffs.directivity)),
            ffs.peakThetaDeg, ffs.peakPhiDeg);
        return;
    }
    if (usingMom)
    {
        if (!momSolver || !momSolver->isFinished())
        {
            MessageDlg(L"Run the MoM solve first.", mtInformation,
                       TMsgDlgButtons() << mbOK, 0);
            return;
        }
        Screen->Cursor = crHourGlass;
        FarFieldData ffm;
        bool okm = momSolver->computeFarField(ffm);
        Screen->Cursor = crDefault;
        if (!okm)
        {
            MessageDlg(L"No far-field data from the MoM run.", mtInformation,
                       TMsgDlgButtons() << mbOK, 0);
            return;
        }
        Aabb b;
        std::vector<Vec3> pts; std::vector<float> mag;
        momSolver->getWireCurrents(pts, mag);
        for (const auto &p : pts) b.grow(p);
        Vec3 center = b.valid() ? b.center() : Vec3(0, 0, 0);
        float scale = b.valid() ? 0.6f * b.size().length() : 1.0f;
        glView->setPattern(ffm, center, scale);
        statusBar->Panels->Items[0]->Text = String().sprintf(
            L"MoM far field:  D = %.2f dBi,  peak at theta=%.0f deg, "
            L"phi=%.0f deg  (pattern radius: 30 dB range).",
            10.0 * std::log10(std::max(1e-6f, ffm.directivity)),
            ffm.peakThetaDeg, ffm.peakPhiDeg);
        return;
    }
    if (usingFem)
    {
        MessageDlg(L"Far-field patterns currently require a TLM/FDTD or MoM "
                   L"run (the FEM solver does not record a Huygens surface "
                   L"yet).", mtInformation, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    if (!solver || !solver->isFinished())
    {
        MessageDlg(L"Run a simulation first.", mtInformation,
                   TMsgDlgButtons() << mbOK, 0);
        return;
    }
    Screen->Cursor = crHourGlass;
    FarFieldData ff;
    bool ok = solver->computeFarField(ff);
    Screen->Cursor = crDefault;
    if (!ok)
    {
        MessageDlg(L"No far-field data (Huygens surface needs padding >= 6 "
                   L"cells and DFT settle time before the run ends).",
                   mtInformation, TMsgDlgButtons() << mbOK, 0);
        return;
    }
    const VoxelGridSpec &g = solver->grid();
    Vec3 center = g.origin + Vec3(g.nx * g.dl, g.ny * g.dl, g.nz * g.dl) * 0.5f;
    float scale = 0.45f * std::min(std::min(g.nx, g.ny), g.nz) * g.dl;
    glView->setPattern(ff, center, scale);
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"Far field @ %.4g GHz:  D = %.2f dBi,  peak at theta=%.0f deg, "
        L"phi=%.0f deg  (pattern radius: 30 dB range)",
        solver->cfg().f0 / 1e9, 10.0 * std::log10(std::max(1e-6f, ff.directivity)),
        ff.peakThetaDeg, ff.peakPhiDeg);
}

//---------------------------------------------------------------------------
// GIF recording
//---------------------------------------------------------------------------
void __fastcall TMainForm::OnRecGifClick(TObject *)
{
    if (recordingGif)
    {
        stopGifRecording();
        return;
    }
    std::unique_ptr<TSaveDialog> dlg(new TSaveDialog(this));
    dlg->Filter = L"Animated GIF (*.gif)|*.gif";
    dlg->DefaultExt = L"gif";
    dlg->FileName = L"rf_simulation.gif";
    if (!dlg->Execute())
        return;
    gifPath = dlg->FileName.c_str();
    recordingGif = true;
    gifStarted = false;
    lastGifStep = -1;
    btnRecGif->Caption = L"■ Stop GIF";
    statusBar->Panels->Items[0]->Text =
        L"Recording GIF: frames are captured while a simulation runs.";
}

//---------------------------------------------------------------------------
void TMainForm::stopGifRecording()
{
    if (!recordingGif)
        return;
    recordingGif = false;
    int n = gif.frameCount();
    gif.finish();
    gifStarted = false;
    btnRecGif->Caption = L"● Record GIF...";
    statusBar->Panels->Items[0]->Text = String().sprintf(
        L"GIF saved: %d frames -> %s", n, String(gifPath.c_str()).c_str());
}

//---------------------------------------------------------------------------
void TMainForm::captureGifFrame()
{
    std::vector<unsigned char> rgb;
    int w = 0, h = 0;
    if (!glView->captureFrame(rgb, w, h) || w < 8 || h < 8)
        return;
    int shrink = std::max(1, (w + 719) / 720);
    if (!gifStarted)
    {
        if (!gif.begin(gifPath, w / shrink, h / shrink, 10))
        {
            recordingGif = false;
            btnRecGif->Caption = L"● Record GIF...";
            return;
        }
        gifStarted = true;
    }
    gif.addFrame(rgb.data(), w, h, shrink);
    if (gif.frameCount() >= 1200)
        stopGifRecording();
}

//---------------------------------------------------------------------------
// playback (frame scrubbing)
//---------------------------------------------------------------------------
void TMainForm::resetPlayback(bool enable)
{
    playing = false;
    playbackMode = false;
    btnPlay->Caption = L"▶ Play";
    int fc = (enable && solver) ? solver->frameCount() : 0;
    updatingSlider = true;
    tbPlayback->Max = std::max(0, fc - 1);
    tbPlayback->Position = tbPlayback->Max;
    updatingSlider = false;
    tbPlayback->Enabled = fc > 1;
    btnPlay->Enabled = fc > 1;
    lblFrame->Caption = (fc > 1)
        ? String().sprintf(L"%d frames recorded - drag to scrub  ", fc)
        : String(L"run a simulation to record frames  ");
}

//---------------------------------------------------------------------------
void TMainForm::showPlaybackFrame(int idx)
{
    if (!solver || !haveGrid)
        return;
    VizFrame fr;
    if (!solver->getFrame(idx, fr))
        return;
    glView->setSurfaceData(solver->faces(), fr.js, lastGrid.dl);
    if (fr.planeAxis >= 0 && !fr.plane.empty())
        glView->setPlaneData(lastGrid, fr.planeAxis, fr.planeIdx,
                             fr.planeN1, fr.planeN2, fr.plane);
    else
        glView->clearPlaneData();
    float tNs = fr.step * solver->timestep() * 1e9f;
    lblFrame->Caption = String().sprintf(
        L"frame %d / %d   step %d   t = %.3f ns  ",
        idx + 1, solver->frameCount(), fr.step, (double)tNs);
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnPlaybackChange(TObject *)
{
    if (updatingSlider || !solver || !solver->isFinished())
        return;
    playbackMode = true;
    showPlaybackFrame(tbPlayback->Position);
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::OnPlayClick(TObject *)
{
    if (!solver || !solver->isFinished() || solver->frameCount() < 2)
        return;
    playing = !playing;
    btnPlay->Caption = playing ? L"❚❚ Pause" : L"▶ Play";
    if (playing)
        playbackMode = true;
}

//---------------------------------------------------------------------------
void __fastcall TMainForm::SaveAs2Click(TObject *Sender)
{
    Close();
}
//---------------------------------------------------------------------------

