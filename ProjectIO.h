//---------------------------------------------------------------------------
// ProjectIO.h - save/load a simulation project (.emsim): the scene, the
// settings, and the computed results so they can be reopened without
// recalculating.
//---------------------------------------------------------------------------
#ifndef ProjectIOH
#define ProjectIOH

#include "Geometry.h"
#include "GLView.h"
#include <string>
#include <vector>
#include <utility>

// One scene object (antenna regenerated from kind+params; STL stores mesh).
struct ProjObject
{
    int    kind = 0;
    std::string name;
    Vec3   position, rotDeg;
    bool   dielectric = false;
    float  epsr = 4.0f, sigma = 0.0f;
    double designFreqHz = 1e9;
    bool   isStl = false;
    std::vector<float> stlVerts;   // 3 per vertex (only if isStl)
    std::vector<int>   stlIdx;
    std::vector<std::pair<std::string, double>> params;
};

// One chart curve / page (mirrors ChartForm)
struct ProjCurve { std::string name; unsigned color = 0; std::vector<double> x, y; };
struct ProjPage  { std::string title, xLabel, yLabel; std::vector<ProjCurve> curves; };

struct ProjectData
{
    std::vector<std::pair<std::string, std::string>> settings;  // vleSim rows
    int  solverIdx = 0, excitationIdx = 0, waveformIdx = 0;
    int  currentsIdx = 1, dbRangeIdx = 2;
    bool gpu = false, showModel = true, dbScale = true;
    bool planeOn = false, meshOn = false;
    std::vector<ProjObject> objects;
    GLResult result;                 // the displayed result snapshot
    std::vector<ProjPage> pages;     // saved plots
};

bool SaveProject(const std::wstring &path, const ProjectData &d);
bool LoadProject(const std::wstring &path, ProjectData &d, std::string &err);

#endif
