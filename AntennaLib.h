//---------------------------------------------------------------------------
// AntennaLib.h - parametric RF component generators
//
// Conventions: boresight along +x, "vertical" polarization along z.
// All dimensions are derived from the design frequency (free-space lambda).
//---------------------------------------------------------------------------
#ifndef AntennaLibH
#define AntennaLibH

#include "Geometry.h"

enum class AntennaKind
{
    Dipole = 0,
    WireEndFed,
    Monopole,
    Yagi,
    LogPeriodic,
    Helix,
    Horn,
    Plate,
    Box,
    Sphere,
    ImportedStl,
    Count
};

const wchar_t *AntennaKindName(AntennaKind k);

// Build a component at the origin. 'params' may be empty -> defaults are
// used; the object's params vector is filled with the values actually used.
SceneObject CreateAntenna(AntennaKind kind, double f0Hz,
                          const std::vector<std::pair<std::string, double>> &params);

double GetParam(const std::vector<std::pair<std::string, double>> &params,
                const char *name, double def);

#endif
