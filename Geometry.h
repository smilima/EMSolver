//---------------------------------------------------------------------------
// Geometry.h - triangle meshes, wires, scene objects, STL import, voxelizer
//---------------------------------------------------------------------------
#ifndef GeometryH
#define GeometryH

#include "Vec3.h"
#include <vector>
#include <string>
#include <cstdint>

//---------------------------------------------------------------------------
// Triangle mesh (indexed). Used both for display and PEC surface voxelization.
//---------------------------------------------------------------------------
struct TriMesh
{
    std::vector<Vec3> verts;
    std::vector<int>  idx;      // 3 indices per triangle
    std::vector<Vec3> normals;  // per-vertex (recomputed by computeNormals)

    int  triCount() const { return (int)idx.size() / 3; }
    void clear() { verts.clear(); idx.clear(); normals.clear(); }
    void addTri(const Vec3 &a, const Vec3 &b, const Vec3 &c);
    void addQuad(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d);
    // Rectangular plate centered at c with edge vectors u and v (full lengths)
    void addPlate(const Vec3 &c, const Vec3 &u, const Vec3 &v);
    void addBox(const Vec3 &lo, const Vec3 &hi);
    void append(const TriMesh &m);
    void transform(const Vec3 &translate); // simple translation
    void computeNormals();
    Aabb bounds() const;
};

//---------------------------------------------------------------------------
// Wire: polyline of points, modeled as thin PEC
//---------------------------------------------------------------------------
struct Wire
{
    std::vector<Vec3> pts;
    float radius = 0.0f;        // 0 -> one cell thick
};

//---------------------------------------------------------------------------
// Feed: excitation gap. E-field is impressed along the segment a->b
// (should span ~1..3 cells). Used for wire/port excitation.
//---------------------------------------------------------------------------
struct Feed
{
    Vec3 a, b;
    bool enabled = false;
};

//---------------------------------------------------------------------------
// A scene object: PEC mesh + PEC wires + optional feed
//---------------------------------------------------------------------------
struct SceneObject
{
    std::string       name;
    TriMesh           mesh;
    std::vector<Wire> wires;
    Feed              feed;
    Vec3              position;         // translation applied to all geometry
    float             designFreqHz = 1e9f; // the frequency it was generated for
    int               kind = 0;          // AntennaKind it was built from (UI use)
    std::vector<std::pair<std::string, double>> params; // editable parameters

    // material: PEC by default; dielectric uses epsr/sigma (solver stubs)
    bool  dielectric = false;
    float epsr  = 4.0f;
    float sigma = 0.0f;

    Aabb worldBounds() const;
};

//---------------------------------------------------------------------------
// STL import (auto-detects ASCII vs binary). Returns false on failure.
//---------------------------------------------------------------------------
bool LoadStl(const std::wstring &path, TriMesh &out, std::string &err);

//---------------------------------------------------------------------------
// Voxelization target: a uniform grid of nx*ny*nz cells with spacing dl,
// cell (i,j,k) spans origin + (i..i+1, j..j+1, k..k+1)*dl.
//---------------------------------------------------------------------------
struct VoxelGridSpec
{
    int  nx = 0, ny = 0, nz = 0;
    Vec3 origin;
    float dl = 1e-3f;

    size_t cellIndex(int i, int j, int k) const
    {
        return ((size_t)k * ny + j) * nx + i;
    }
    bool inGrid(int i, int j, int k) const
    {
        return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz;
    }
    // world position of cell center
    Vec3 cellCenter(int i, int j, int k) const
    {
        return Vec3(origin.x + (i + 0.5f) * dl,
                    origin.y + (j + 0.5f) * dl,
                    origin.z + (k + 0.5f) * dl);
    }
    // cell containing world point p (no clamping)
    void cellOf(const Vec3 &p, int &i, int &j, int &k) const
    {
        i = (int)std::floor((p.x - origin.x) / dl);
        j = (int)std::floor((p.y - origin.y) / dl);
        k = (int)std::floor((p.z - origin.z) / dl);
    }
};

// Mark cells overlapped by mesh triangles (surface voxelization) by setting
// mat[cell] = matId. Mesh is translated by 'offset' first.
void VoxelizeMesh(const TriMesh &mesh, const Vec3 &offset,
                  const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                  uint8_t matId);

// Solid (filled) voxelization of a closed mesh: x-ray parity fill plus a
// surface pass. Used for dielectric volumes.
void VoxelizeMeshSolid(const TriMesh &mesh, const Vec3 &offset,
                       const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                       uint8_t matId);

// Rasterize a wire polyline into cells (3D DDA, with radius dilation).
void VoxelizeWire(const Wire &w, const Vec3 &offset,
                  const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                  uint8_t matId);

// Triangle / axis-aligned-box overlap test (separating axis theorem)
bool TriBoxOverlap(const Vec3 &boxCenter, const Vec3 &boxHalf,
                   const Vec3 &v0, const Vec3 &v1, const Vec3 &v2);

//---------------------------------------------------------------------------
// A PEC surface face (between a PEC cell and an adjacent non-PEC cell).
// Shared by the TLM and FEM solvers; surface current Js = n x H is sampled
// at the adjacent cell.
//---------------------------------------------------------------------------
struct SurfaceFace
{
    int    airCell;     // linear index of the adjacent non-PEC cell
    Vec3   center;      // world position of face center (for rendering)
    int8_t axis;        // face normal axis 0..2
    int8_t sign;        // +1: air is on + side of the PEC cell, -1: - side
};

// Build the PEC surface-face list from a voxel material map
// (matPec is the PEC material id, normally MAT_PEC == 1).
void BuildPecSurfaceFaces(const VoxelGridSpec &g,
                          const std::vector<uint8_t> &mat, uint8_t matPec,
                          std::vector<SurfaceFace> &out);

// Cube-edge wireframe (point pairs) of all non-air voxels: shows the
// staircased TLM/FDTD discretization of the geometry. Edges are
// deduplicated; output is capped at maxSegments.
void BuildVoxelOutline(const VoxelGridSpec &g,
                       const std::vector<uint8_t> &mat,
                       std::vector<Vec3> &segments, size_t maxSegments);

#endif
