//---------------------------------------------------------------------------
// AntennaLib.cpp - parametric RF component generators
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "AntennaLib.h"
#include <cmath>

#pragma package(smart_init)

static const double C0 = 299792458.0;

const wchar_t *AntennaKindName(AntennaKind k)
{
    switch (k)
    {
    case AntennaKind::Dipole:      return L"Dipole";
    case AntennaKind::WireEndFed:  return L"Wire (end fed)";
    case AntennaKind::Monopole:    return L"Monopole + ground";
    case AntennaKind::Yagi:        return L"Yagi-Uda";
    case AntennaKind::LogPeriodic: return L"Log-periodic (LPDA)";
    case AntennaKind::Helix:       return L"Helical (axial mode)";
    case AntennaKind::Horn:        return L"Pyramidal horn";
    case AntennaKind::Plate:       return L"PEC plate";
    case AntennaKind::Box:         return L"PEC box";
    case AntennaKind::Sphere:      return L"PEC sphere";
    case AntennaKind::ImportedStl: return L"Imported STL";
    default:                       return L"Component";
    }
}

//---------------------------------------------------------------------------
double GetParam(const std::vector<std::pair<std::string, double>> &params,
                const char *name, double def)
{
    for (const auto &p : params)
        if (p.first == name)
            return p.second;
    return def;
}

//---------------------------------------------------------------------------
static void SetParam(SceneObject &o, const char *name, double v)
{
    o.params.push_back({ name, v });
}

//---------------------------------------------------------------------------
static Wire MakeSegment(const Vec3 &a, const Vec3 &b)
{
    Wire w;
    w.pts.push_back(a);
    w.pts.push_back(b);
    return w;
}

//---------------------------------------------------------------------------
static void BuildDipole(SceneObject &o, double lam,
                        const std::vector<std::pair<std::string, double>> &p)
{
    double lenF = GetParam(p, "Length/lambda", 0.475);
    float L = (float)(lenF * lam), g = (float)(lam / 25.0);
    o.wires.push_back(MakeSegment(Vec3(0, 0, g / 2),  Vec3(0, 0, L / 2)));
    o.wires.push_back(MakeSegment(Vec3(0, 0, -g / 2), Vec3(0, 0, -L / 2)));
    o.feed = { Vec3(0, 0, -g / 2), Vec3(0, 0, g / 2), true };
    SetParam(o, "Length/lambda", lenF);
}

//---------------------------------------------------------------------------
static void BuildWireEndFed(SceneObject &o, double lam,
                            const std::vector<std::pair<std::string, double>> &p)
{
    double lenF = GetParam(p, "Length/lambda", 1.0);
    float L = (float)(lenF * lam), g = (float)(lam / 25.0);
    o.wires.push_back(MakeSegment(Vec3(g, 0, 0), Vec3(g + L, 0, 0)));
    o.feed = { Vec3(0, 0, 0), Vec3(g, 0, 0), true };
    SetParam(o, "Length/lambda", lenF);
}

//---------------------------------------------------------------------------
static void BuildMonopole(SceneObject &o, double lam,
                          const std::vector<std::pair<std::string, double>> &p)
{
    double lenF = GetParam(p, "Length/lambda", 0.24);
    double gpF  = GetParam(p, "Ground/lambda", 1.0);
    float L = (float)(lenF * lam), g = (float)(lam / 25.0);
    float gp = (float)(gpF * lam);
    o.mesh.addPlate(Vec3(0, 0, 0), Vec3(gp, 0, 0), Vec3(0, gp, 0));
    o.wires.push_back(MakeSegment(Vec3(0, 0, g), Vec3(0, 0, g + L)));
    o.feed = { Vec3(0, 0, 0), Vec3(0, 0, g), true };
    SetParam(o, "Length/lambda", lenF);
    SetParam(o, "Ground/lambda", gpF);
}

//---------------------------------------------------------------------------
static void BuildYagi(SceneObject &o, double lam,
                      const std::vector<std::pair<std::string, double>> &p)
{
    int nDir = (int)GetParam(p, "Directors", 4);
    if (nDir < 0)  nDir = 0;
    if (nDir > 20) nDir = 20;
    float g = (float)(lam / 25.0);

    auto element = [&](double x, double len)
    {
        o.wires.push_back(MakeSegment(Vec3((float)x, 0, (float)(-len / 2)),
                                      Vec3((float)x, 0, (float)(len / 2))));
    };
    // reflector
    element(0.0, 0.495 * lam);
    // driven element (split, fed at center)
    double xd = 0.20 * lam, Ld = 0.473 * lam;
    o.wires.push_back(MakeSegment(Vec3((float)xd, 0, g / 2),
                                  Vec3((float)xd, 0, (float)(Ld / 2))));
    o.wires.push_back(MakeSegment(Vec3((float)xd, 0, -g / 2),
                                  Vec3((float)xd, 0, (float)(-Ld / 2))));
    o.feed = { Vec3((float)xd, 0, -g / 2), Vec3((float)xd, 0, g / 2), true };
    // directors
    double x = xd + 0.25 * lam;
    for (int d = 0; d < nDir; ++d)
    {
        double len = (0.440 - 0.005 * d) * lam;
        element(x, len);
        x += 0.31 * lam;
    }
    SetParam(o, "Directors", nDir);
}

//---------------------------------------------------------------------------
static void BuildLogPeriodic(SceneObject &o, double lam,
                             const std::vector<std::pair<std::string, double>> &p)
{
    int    N     = (int)GetParam(p, "Elements", 8);
    double tau   = GetParam(p, "Tau", 0.88);
    double sigma = GetParam(p, "Sigma", 0.12);
    if (N < 3)  N = 3;
    if (N > 20) N = 20;
    if (tau < 0.7)  tau = 0.7;
    if (tau > 0.98) tau = 0.98;

    double s  = 0.03 * lam;              // boom separation (y)
    double L1 = 0.5 * lam / 0.7;         // longest element, covers 0.7*f0
    double x  = 0.0;
    double yA = s / 2, yB = -s / 2;

    double xPrev = 0.0;
    for (int n = 0; n < N; ++n)
    {
        double L = L1 * std::pow(tau, n);
        // criss-cross: alternate which boom feeds which half
        double yTop = (n % 2 == 0) ? yA : yB;
        double yBot = (n % 2 == 0) ? yB : yA;
        o.wires.push_back(MakeSegment(Vec3((float)x, (float)yTop, 0),
                                      Vec3((float)x, (float)yTop, (float)(L / 2))));
        o.wires.push_back(MakeSegment(Vec3((float)x, (float)yBot, 0),
                                      Vec3((float)x, (float)yBot, (float)(-L / 2))));
        xPrev = x;
        x += 2.0 * sigma * L;
    }
    // booms (two parallel wires along x), front extension to the feed
    double xf = xPrev + 0.04 * lam;
    o.wires.push_back(MakeSegment(Vec3(0, (float)yA, 0), Vec3((float)xf, (float)yA, 0)));
    o.wires.push_back(MakeSegment(Vec3(0, (float)yB, 0), Vec3((float)xf, (float)yB, 0)));
    // feed across the booms at the front (short-element end)
    o.feed = { Vec3((float)xf, (float)yB, 0), Vec3((float)xf, (float)yA, 0), true };
    SetParam(o, "Elements", N);
    SetParam(o, "Tau", tau);
    SetParam(o, "Sigma", sigma);
}

//---------------------------------------------------------------------------
static void BuildHelix(SceneObject &o, double lam,
                       const std::vector<std::pair<std::string, double>> &p)
{
    double turns = GetParam(p, "Turns", 8);
    double circF = GetParam(p, "Circumference/lambda", 1.0);
    double pitchF = GetParam(p, "Pitch/lambda", 0.24);
    if (turns < 1)  turns = 1;
    if (turns > 40) turns = 40;

    double r = circF * lam / (2.0 * M_PI);
    double S = pitchF * lam;             // axial advance per turn
    float  g = (float)(lam / 25.0);
    double gp = 0.9 * lam;               // ground plane edge

    o.mesh.addPlate(Vec3(0, 0, 0), Vec3(0, (float)gp, 0), Vec3(0, 0, (float)gp));

    Wire w;
    int segsPerTurn = 24;
    int total = (int)(turns * segsPerTurn + 0.5);
    for (int i = 0; i <= total; ++i)
    {
        double t = 2.0 * M_PI * i / segsPerTurn;
        w.pts.push_back(Vec3((float)(g + S * t / (2.0 * M_PI)),
                             (float)(r * std::cos(t)),
                             (float)(r * std::sin(t))));
    }
    o.wires.push_back(w);
    o.feed = { Vec3(0, (float)r, 0), Vec3(g, (float)r, 0), true };
    SetParam(o, "Turns", turns);
    SetParam(o, "Circumference/lambda", circF);
    SetParam(o, "Pitch/lambda", pitchF);
}

//---------------------------------------------------------------------------
static void BuildHorn(SceneObject &o, double lam,
                      const std::vector<std::pair<std::string, double>> &p)
{
    double Af = GetParam(p, "ApertureW/lambda", 1.8);   // y
    double Bf = GetParam(p, "ApertureH/lambda", 1.35);  // z
    double Ff = GetParam(p, "FlareLen/lambda", 1.8);

    double a = 0.70 * lam, b = 0.32 * lam;     // waveguide cross-section
    double Lwg = 0.80 * lam;
    double A = Af * lam, B = Bf * lam, Lf = Ff * lam;
    float g = (float)(lam / 25.0);

    auto P = [](double x, double y, double z)
    {
        return Vec3((float)x, (float)y, (float)z);
    };

    TriMesh &m = o.mesh;
    // back wall (x = 0)
    m.addPlate(P(0, 0, 0), P(0, a, 0), P(0, 0, b));
    // waveguide walls
    m.addQuad(P(0, -a/2, -b/2), P(Lwg, -a/2, -b/2), P(Lwg, -a/2, b/2), P(0, -a/2, b/2)); // -y
    m.addQuad(P(0,  a/2, -b/2), P(Lwg,  a/2, -b/2), P(Lwg,  a/2, b/2), P(0,  a/2, b/2)); // +y
    m.addQuad(P(0, -a/2, -b/2), P(Lwg, -a/2, -b/2), P(Lwg, a/2, -b/2), P(0, a/2, -b/2)); // -z
    m.addQuad(P(0, -a/2,  b/2), P(Lwg, -a/2,  b/2), P(Lwg, a/2,  b/2), P(0, a/2,  b/2)); // +z
    // flare walls (waveguide mouth -> aperture)
    double xe = Lwg + Lf;
    m.addQuad(P(Lwg, -a/2, -b/2), P(xe, -A/2, -B/2), P(xe, -A/2, B/2), P(Lwg, -a/2, b/2)); // -y
    m.addQuad(P(Lwg,  a/2, -b/2), P(xe,  A/2, -B/2), P(xe,  A/2, B/2), P(Lwg,  a/2, b/2)); // +y
    m.addQuad(P(Lwg, -a/2, -b/2), P(xe, -A/2, -B/2), P(xe, A/2, -B/2), P(Lwg, a/2, -b/2)); // -z
    m.addQuad(P(Lwg, -a/2,  b/2), P(xe, -A/2,  B/2), P(xe, A/2,  B/2), P(Lwg, a/2,  b/2)); // +z

    // TE10 probe: quarter guide wavelength from the back short, E along z
    double ratio = lam / (2.0 * a);
    double lamG  = lam / std::sqrt(std::max(0.05, 1.0 - ratio * ratio));
    double xp = 0.25 * lamG;
    if (xp > Lwg - 0.05 * lam)
        xp = Lwg * 0.5;
    double probeLen = 0.20 * lam;
    o.wires.push_back(MakeSegment(P(xp, 0, -b/2 + g), P(xp, 0, -b/2 + g + probeLen)));
    o.feed = { P(xp, 0, -b/2), P(xp, 0, -b/2 + g), true };

    SetParam(o, "ApertureW/lambda", Af);
    SetParam(o, "ApertureH/lambda", Bf);
    SetParam(o, "FlareLen/lambda", Ff);
}

//---------------------------------------------------------------------------
// Material parameters shared by the passive solids (plate/box/sphere):
// "Dielectric (0/1)" selects PEC vs dielectric, with "Epsr"/"Sigma (S/m)".
static void ApplyMaterialParams(SceneObject &o,
                                const std::vector<std::pair<std::string, double>> &p)
{
    o.dielectric = GetParam(p, "Dielectric (0/1)", 0.0) >= 0.5;
    o.epsr  = (float)GetParam(p, "Epsr", 4.0);
    o.sigma = (float)GetParam(p, "Sigma (S/m)", 0.0);
    if (o.epsr < 1.0f) o.epsr = 1.0f;
    if (o.sigma < 0.0f) o.sigma = 0.0f;
    SetParam(o, "Dielectric (0/1)", o.dielectric ? 1.0 : 0.0);
    SetParam(o, "Epsr", o.epsr);
    SetParam(o, "Sigma (S/m)", o.sigma);
}

//---------------------------------------------------------------------------
static void BuildPlate(SceneObject &o, double lam,
                       const std::vector<std::pair<std::string, double>> &p)
{
    double w = GetParam(p, "Width/lambda", 1.0);
    double h = GetParam(p, "Height/lambda", 1.0);
    o.mesh.addPlate(Vec3(0, 0, 0), Vec3(0, (float)(w * lam), 0),
                    Vec3(0, 0, (float)(h * lam)));
    SetParam(o, "Width/lambda", w);
    SetParam(o, "Height/lambda", h);
    ApplyMaterialParams(o, p);
}

//---------------------------------------------------------------------------
static void BuildBox(SceneObject &o, double lam,
                     const std::vector<std::pair<std::string, double>> &p)
{
    double sx = GetParam(p, "SizeX/lambda", 0.5);
    double sy = GetParam(p, "SizeY/lambda", 0.5);
    double sz = GetParam(p, "SizeZ/lambda", 0.5);
    Vec3 h((float)(sx * lam / 2), (float)(sy * lam / 2), (float)(sz * lam / 2));
    o.mesh.addBox(-h, h);
    SetParam(o, "SizeX/lambda", sx);
    SetParam(o, "SizeY/lambda", sy);
    SetParam(o, "SizeZ/lambda", sz);
    ApplyMaterialParams(o, p);
}

//---------------------------------------------------------------------------
static void BuildSphere(SceneObject &o, double lam,
                        const std::vector<std::pair<std::string, double>> &p)
{
    double dF = GetParam(p, "Diameter/lambda", 1.0);
    float r = (float)(dF * lam / 2);
    const int nu = 28, nv = 14;
    for (int v = 0; v < nv; ++v)
    {
        double th0 = M_PI * v / nv, th1 = M_PI * (v + 1) / nv;
        for (int u = 0; u < nu; ++u)
        {
            double ph0 = 2 * M_PI * u / nu, ph1 = 2 * M_PI * (u + 1) / nu;
            auto sp = [&](double th, double ph)
            {
                return Vec3((float)(r * std::cos(th)),
                            (float)(r * std::sin(th) * std::cos(ph)),
                            (float)(r * std::sin(th) * std::sin(ph)));
            };
            o.mesh.addQuad(sp(th0, ph0), sp(th0, ph1), sp(th1, ph1), sp(th1, ph0));
        }
    }
    SetParam(o, "Diameter/lambda", dF);
    ApplyMaterialParams(o, p);
}

//---------------------------------------------------------------------------
SceneObject CreateAntenna(AntennaKind kind, double f0Hz,
                          const std::vector<std::pair<std::string, double>> &params)
{
    SceneObject o;
    double lam = C0 / f0Hz;
    o.designFreqHz = (float)f0Hz;
    o.kind = (int)kind;

    switch (kind)
    {
    case AntennaKind::Dipole:      BuildDipole(o, lam, params); break;
    case AntennaKind::WireEndFed:  BuildWireEndFed(o, lam, params); break;
    case AntennaKind::Monopole:    BuildMonopole(o, lam, params); break;
    case AntennaKind::Yagi:        BuildYagi(o, lam, params); break;
    case AntennaKind::LogPeriodic: BuildLogPeriodic(o, lam, params); break;
    case AntennaKind::Helix:       BuildHelix(o, lam, params); break;
    case AntennaKind::Horn:        BuildHorn(o, lam, params); break;
    case AntennaKind::Plate:       BuildPlate(o, lam, params); break;
    case AntennaKind::Box:         BuildBox(o, lam, params); break;
    case AntennaKind::Sphere:      BuildSphere(o, lam, params); break;
    default: break;
    }
    o.mesh.computeNormals();

    char nameBuf[64];
    AnsiString an(AntennaKindName(kind));
    sprintf(nameBuf, "%s", an.c_str());
    o.name = nameBuf;
    return o;
}
