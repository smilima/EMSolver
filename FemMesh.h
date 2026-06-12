//---------------------------------------------------------------------------
// FemMesh.h - tetrahedral mesh generator for the FEM solver
//
// v1 strategy: structured mesh derived from the voxel grid. Every non-PEC
// cell is split into 6 Kuhn tetrahedra (all cells split identically, which
// guarantees a conforming mesh). PEC volumes are excluded from the mesh;
// their surfaces become tangential-E = 0 (PEC) edge constraints. Materials
// (epsr, sigma) are carried per tetrahedron from the voxel map.
//---------------------------------------------------------------------------
#ifndef FemMeshH
#define FemMeshH

#include "Geometry.h"
#include <vector>
#include <cstdint>

struct FemTet
{
    int     n[4];       // node indices
    int     edge[6];    // global edge ids, local edges (01,02,03,12,13,23)
    int8_t  esign[6];   // +1 if local orientation == global (lo -> hi node)
    int     cell;       // owning voxel cell (linear index)
    uint8_t mat;        // voxel material id
};

// outer-boundary triangle (for the absorbing boundary condition)
struct FemBFace
{
    int    n[3];
    int    e[3];        // global edge ids of (n0n1, n0n2, n1n2)
    int8_t s[3];
};

class FemMesh
{
public:
    // returns false if the grid is empty / degenerate
    bool build(const VoxelGridSpec &grid, const std::vector<uint8_t> &mat);

    const VoxelGridSpec &grid() const { return g; }

    std::vector<Vec3>    nodes;
    std::vector<FemTet>  tets;
    std::vector<std::pair<int, int>> edges;    // (loNode, hiNode)
    std::vector<uint8_t> edgePec;              // 1 = tangential-E constrained
    std::vector<FemBFace> bfaces;              // outer boundary triangles

    // the 4 pol-axis "pillar" edges of a voxel cell (for gap excitation);
    // returns global edge ids (orientation is always +polAxis)
    void cellPillarEdges(size_t cell, int polAxis, std::vector<int> &out) const;

    // wireframe segments (point pairs) of tets in PEC-adjacent cells, for
    // mesh visualization
    void vizEdges(const std::vector<uint8_t> &mat,
                  std::vector<Vec3> &segments, size_t maxSegments) const;

private:
    VoxelGridSpec g;
    std::vector<int> nodeId;       // (nx+1)*(ny+1)*(nz+1) -> node index or -1

    size_t nodeKey(int i, int j, int k) const
    {
        return ((size_t)k * (g.ny + 1) + j) * (g.nx + 1) + i;
    }
    int findEdge(int a, int b) const;          // -1 if absent

    std::vector<int>  edgeHash;                // open hash: head per bucket
    std::vector<int>  edgeNext;                // chained list
    int  edgeLookup(int a, int b);             // find or create
    int  hashSize = 0;
};

#endif
