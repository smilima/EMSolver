//---------------------------------------------------------------------------
// ProjectIO.cpp - versioned binary serialization of a project + results
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "ProjectIO.h"
#include <cstdio>
#include <cstdint>

#pragma package(smart_init)

static const char     PROJ_MAGIC[8] = { 'E','M','S','I','M','\0','\0','\0' };
static const uint32_t PROJ_VERSION  = 2;   // v2 adds recorded playback frames

//---------------------------------------------------------------------------
namespace {

struct Writer
{
    FILE *f; bool ok = true;
    void raw(const void *p, size_t n) { if (ok && fwrite(p, 1, n, f) != n) ok = false; }
    template<class T> void pod(const T &v) { raw(&v, sizeof(T)); }
    void str(const std::string &s)
    {
        uint32_t n = (uint32_t)s.size(); pod(n); raw(s.data(), n);
    }
    template<class T> void vpod(const std::vector<T> &v)
    {
        uint32_t n = (uint32_t)v.size(); pod(n);
        if (n) raw(v.data(), (size_t)n * sizeof(T));
    }
    void params(const std::vector<std::pair<std::string, double>> &p)
    {
        uint32_t n = (uint32_t)p.size(); pod(n);
        for (auto &kv : p) { str(kv.first); pod(kv.second); }
    }
};

struct Reader
{
    FILE *f; bool ok = true;
    void raw(void *p, size_t n) { if (ok && fread(p, 1, n, f) != n) ok = false; }
    template<class T> void pod(T &v) { raw(&v, sizeof(T)); }
    std::string str()
    {
        uint32_t n = 0; pod(n);
        std::string s;
        if (ok && n < 200000000u) { s.resize(n); if (n) raw(&s[0], n); }
        else ok = false;
        return s;
    }
    template<class T> void vpod(std::vector<T> &v)
    {
        uint32_t n = 0; pod(n);
        if (ok && (size_t)n * sizeof(T) < 4000000000u)
        { v.resize(n); if (n) raw(v.data(), (size_t)n * sizeof(T)); }
        else ok = false;
    }
    void params(std::vector<std::pair<std::string, double>> &p)
    {
        uint32_t n = 0; pod(n);
        if (n > 100000) { ok = false; return; }
        p.resize(n);
        for (auto &kv : p) { kv.first = str(); pod(kv.second); }
    }
};

//---------------------------------------------------------------------------
void wGLResult(Writer &w, const GLResult &r)
{
    w.vpod(r.faces); w.vpod(r.faceVals); w.pod(r.faceDl);
    w.vpod(r.triV); w.vpod(r.triIdx); w.vpod(r.triMag);
    w.vpod(r.wirePts); w.vpod(r.wireMag);
    uint8_t hp = r.hasPlane ? 1 : 0; w.pod(hp);
    w.pod(r.planeGrid); w.pod(r.planeAxis); w.pod(r.planeIdx);
    w.pod(r.planeN1); w.pod(r.planeN2); w.vpod(r.planeVals);
    uint8_t hpat = r.hasPattern ? 1 : 0; w.pod(hpat);
    w.pod(r.pattern.nTheta); w.pod(r.pattern.nPhi); w.vpod(r.pattern.U);
    w.pod(r.pattern.uMax); w.pod(r.pattern.directivity);
    w.pod(r.pattern.peakThetaDeg); w.pod(r.pattern.peakPhiDeg);
    w.pod(r.patCenter); w.pod(r.patScale);
    w.pod(r.domain); uint8_t dv = r.domainVisible ? 1 : 0; w.pod(dv);
    w.vpod(r.probeMarkers);
}

void rGLResult(Reader &r, GLResult &g)
{
    r.vpod(g.faces); r.vpod(g.faceVals); r.pod(g.faceDl);
    r.vpod(g.triV); r.vpod(g.triIdx); r.vpod(g.triMag);
    r.vpod(g.wirePts); r.vpod(g.wireMag);
    uint8_t hp = 0; r.pod(hp); g.hasPlane = hp != 0;
    r.pod(g.planeGrid); r.pod(g.planeAxis); r.pod(g.planeIdx);
    r.pod(g.planeN1); r.pod(g.planeN2); r.vpod(g.planeVals);
    uint8_t hpat = 0; r.pod(hpat); g.hasPattern = hpat != 0;
    r.pod(g.pattern.nTheta); r.pod(g.pattern.nPhi); r.vpod(g.pattern.U);
    r.pod(g.pattern.uMax); r.pod(g.pattern.directivity);
    r.pod(g.pattern.peakThetaDeg); r.pod(g.pattern.peakPhiDeg);
    r.pod(g.patCenter); r.pod(g.patScale);
    r.pod(g.domain); uint8_t dv = 0; r.pod(dv); g.domainVisible = dv != 0;
    r.vpod(g.probeMarkers);
}

//---------------------------------------------------------------------------
void wVizFrame(Writer &w, const VizFrame &fr)
{
    w.pod(fr.step); w.pod(fr.planeAxis); w.pod(fr.planeIdx);
    w.pod(fr.planeN1); w.pod(fr.planeN2);
    w.vpod(fr.js); w.vpod(fr.plane);
}

void rVizFrame(Reader &r, VizFrame &fr)
{
    r.pod(fr.step); r.pod(fr.planeAxis); r.pod(fr.planeIdx);
    r.pod(fr.planeN1); r.pod(fr.planeN2);
    r.vpod(fr.js); r.vpod(fr.plane);
}

} // namespace

//---------------------------------------------------------------------------
bool SaveProject(const std::wstring &path, const ProjectData &d)
{
    FILE *f = _wfopen(path.c_str(), L"wb");
    if (!f) return false;
    Writer w; w.f = f;
    w.raw(PROJ_MAGIC, 8);
    w.pod(PROJ_VERSION);

    // settings
    uint32_t ns = (uint32_t)d.settings.size(); w.pod(ns);
    for (auto &kv : d.settings) { w.str(kv.first); w.str(kv.second); }
    // toolbar
    w.pod(d.solverIdx); w.pod(d.excitationIdx); w.pod(d.waveformIdx);
    w.pod(d.currentsIdx); w.pod(d.dbRangeIdx);
    uint8_t b[5] = { (uint8_t)d.gpu, (uint8_t)d.showModel, (uint8_t)d.dbScale,
                     (uint8_t)d.planeOn, (uint8_t)d.meshOn };
    w.raw(b, 5);
    // objects
    uint32_t no = (uint32_t)d.objects.size(); w.pod(no);
    for (const auto &o : d.objects)
    {
        w.pod(o.kind); w.str(o.name);
        w.pod(o.position); w.pod(o.rotDeg);
        uint8_t di = o.dielectric ? 1 : 0; w.pod(di);
        w.pod(o.epsr); w.pod(o.sigma); w.pod(o.designFreqHz);
        uint8_t st = o.isStl ? 1 : 0; w.pod(st);
        w.vpod(o.stlVerts); w.vpod(o.stlIdx);
        w.params(o.params);
    }
    // result + plots
    wGLResult(w, d.result);
    uint32_t npg = (uint32_t)d.pages.size(); w.pod(npg);
    for (const auto &pg : d.pages)
    {
        w.str(pg.title); w.str(pg.xLabel); w.str(pg.yLabel);
        uint32_t nc = (uint32_t)pg.curves.size(); w.pod(nc);
        for (const auto &c : pg.curves)
        {
            w.str(c.name); w.pod(c.color); w.vpod(c.x); w.vpod(c.y);
        }
    }
    // playback frames (v2)
    uint8_t hpb = d.playback.valid ? 1 : 0; w.pod(hpb);
    if (hpb)
    {
        w.pod(d.playback.grid); w.pod(d.playback.dt);
        w.vpod(d.playback.faces);
        uint32_t nf = (uint32_t)d.playback.frames.size(); w.pod(nf);
        for (const auto &fr : d.playback.frames)
            wVizFrame(w, fr);
    }
    bool ok = w.ok;
    fclose(f);
    return ok;
}

//---------------------------------------------------------------------------
bool LoadProject(const std::wstring &path, ProjectData &d, std::string &err)
{
    FILE *f = _wfopen(path.c_str(), L"rb");
    if (!f) { err = "cannot open file"; return false; }
    Reader r; r.f = f;
    char magic[8]; r.raw(magic, 8);
    if (!r.ok || memcmp(magic, PROJ_MAGIC, 8) != 0)
    { fclose(f); err = "not an EMSIM project file"; return false; }
    uint32_t ver = 0; r.pod(ver);
    if (ver < 1 || ver > PROJ_VERSION)
    { fclose(f); err = "unsupported project version"; return false; }

    d = ProjectData();
    uint32_t ns = 0; r.pod(ns);
    if (ns > 1000) { fclose(f); err = "corrupt settings"; return false; }
    d.settings.resize(ns);
    for (auto &kv : d.settings) { kv.first = r.str(); kv.second = r.str(); }
    r.pod(d.solverIdx); r.pod(d.excitationIdx); r.pod(d.waveformIdx);
    r.pod(d.currentsIdx); r.pod(d.dbRangeIdx);
    uint8_t b[5] = {0}; r.raw(b, 5);
    d.gpu = b[0]; d.showModel = b[1]; d.dbScale = b[2];
    d.planeOn = b[3]; d.meshOn = b[4];
    uint32_t no = 0; r.pod(no);
    if (no > 100000) { fclose(f); err = "corrupt object list"; return false; }
    d.objects.resize(no);
    for (auto &o : d.objects)
    {
        r.pod(o.kind); o.name = r.str();
        r.pod(o.position); r.pod(o.rotDeg);
        uint8_t di = 0; r.pod(di); o.dielectric = di != 0;
        r.pod(o.epsr); r.pod(o.sigma); r.pod(o.designFreqHz);
        uint8_t st = 0; r.pod(st); o.isStl = st != 0;
        r.vpod(o.stlVerts); r.vpod(o.stlIdx);
        r.params(o.params);
    }
    rGLResult(r, d.result);
    uint32_t npg = 0; r.pod(npg);
    if (npg > 1000) { fclose(f); err = "corrupt plot data"; return false; }
    d.pages.resize(npg);
    for (auto &pg : d.pages)
    {
        pg.title = r.str(); pg.xLabel = r.str(); pg.yLabel = r.str();
        uint32_t nc = 0; r.pod(nc);
        if (nc > 1000) { fclose(f); err = "corrupt curve data"; return false; }
        pg.curves.resize(nc);
        for (auto &c : pg.curves)
        { c.name = r.str(); r.pod(c.color); r.vpod(c.x); r.vpod(c.y); }
    }
    // playback frames (v2+)
    if (ver >= 2)
    {
        uint8_t hpb = 0; r.pod(hpb);
        if (hpb)
        {
            d.playback.valid = true;
            r.pod(d.playback.grid); r.pod(d.playback.dt);
            r.vpod(d.playback.faces);
            uint32_t nf = 0; r.pod(nf);
            if (nf > 10000000u)
            { fclose(f); err = "corrupt playback data"; return false; }
            d.playback.frames.resize(nf);
            for (auto &fr : d.playback.frames)
                rVizFrame(r, fr);
        }
    }
    bool ok = r.ok;
    fclose(f);
    if (!ok) err = "file truncated or corrupt";
    return ok;
}
