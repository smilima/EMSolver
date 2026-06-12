//---------------------------------------------------------------------------
// FemMesh.cpp - voxel-based Kuhn tetrahedral mesh generator
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "FemMesh.h"
#include "TlmSolver.h"      // material id constants
#include <algorithm>
#include <cstring>

#pragma package(smart_init)

// Kuhn subdivision: 6 tets per cube, all sharing the main diagonal
// 000 -> 111. Each tet visits corners along one axis permutation.
static const int KUHN[6][3] = {
    {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
};

//---------------------------------------------------------------------------
int FemMesh::edgeLookup(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    uint64_t h = ((uint64_t)a * 2654435761u ^ (uint64_t)b * 40503u) % hashSize;
    for (int e = edgeHash[h]; e >= 0; e = edgeNext[e])
        if (edges[e].first == a && edges[e].second == b)
            return e;
    int id = (int)edges.size();
    edges.push_back({ a, b });
    edgeNext.push_back(edgeHash[h]);
    edgeHash[h] = id;
    return id;
}

//---------------------------------------------------------------------------
int FemMesh::findEdge(int a, int b) const
{
    if (a > b)
        std::swap(a, b);
    uint64_t h = ((uint64_t)a * 2654435761u ^ (uint64_t)b * 40503u) % hashSize;
    for (int e = edgeHash[h]; e >= 0; e = edgeNext[e])
        if (edges[e].first == a && edges[e].second == b)
            return e;
    return -1;
}

//---------------------------------------------------------------------------
bool FemMesh::build(const VoxelGridSpec &grid, const std::vector<uint8_t> &mat)
{
    g = grid;
    nodes.clear();
    tets.clear();
    edges.clear();
    edgePec.clear();
    bfaces.clear();
    if (g.nx < 2 || g.ny < 2 || g.nz < 2)
        return false;

    const size_t nNodeSlots = (size_t)(g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    nodeId.assign(nNodeSlots, -1);

    // count non-PEC cells for sizing
    size_t nAir = 0;
    for (uint8_t m : mat)
        if (m != MAT_PEC)
            ++nAir;
    if (nAir == 0)
        return false;
    tets.reserve(nAir * 6);
    edges.reserve(nAir * 8);
    hashSize = (int)std::max<size_t>(1024, nAir * 8);
    edgeHash.assign(hashSize, -1);
    edgeNext.clear();
    edgeNext.reserve(nAir * 8);

    auto getNode = [&](int i, int j, int k) -> int
    {
        size_t key = nodeKey(i, j, k);
        int id = nodeId[key];
        if (id < 0)
        {
            id = (int)nodes.size();
            nodeId[key] = id;
            nodes.push_back(Vec3(g.origin.x + i * g.dl,
                                 g.origin.y + j * g.dl,
                                 g.origin.z + k * g.dl));
        }
        return id;
    };

    // ---- tetrahedralize all non-PEC cells ----
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                size_t c = g.cellIndex(i, j, k);
                if (mat[c] == MAT_PEC)
                    continue;
                int corner[2][2][2];
                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            corner[dx][dy][dz] = getNode(i + dx, j + dy, k + dz);

                for (int t = 0; t < 6; ++t)
                {
                    int d[3] = { 0, 0, 0 };
                    FemTet tet;
                    tet.cell = (int)c;
                    tet.mat  = mat[c];
                    tet.n[0] = corner[0][0][0];
                    for (int s = 0; s < 3; ++s)
                    {
                        d[KUHN[t][s]] = 1;
                        tet.n[s + 1] = corner[d[0]][d[1]][d[2]];
                    }
                    // edges, local order (01,02,03,12,13,23)
                    static const int EP[6][2] = {
                        {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
                    for (int e = 0; e < 6; ++e)
                    {
                        int a = tet.n[EP[e][0]], b = tet.n[EP[e][1]];
                        tet.edge[e]  = edgeLookup(a, b);
                        tet.esign[e] = (a < b) ? 1 : -1;
                    }
                    tets.push_back(tet);
                }
            }

    // ---- PEC edge constraints ----
    // An edge is PEC if some PEC voxel's closed cube contains both of its
    // endpoints (then the whole straight edge lies in/on that conductor).
    edgePec.assign(edges.size(), 0);
    auto cellsTouching = [&](const Vec3 &p, int io[2], int jo[2], int ko[2])
    {
        // integer cell ranges whose closed cube contains p
        auto rng = [&](float x, float o, int n, int out[2])
        {
            float t = (x - o) / g.dl;
            int lo = (int)std::floor(t - 1e-4f);
            int hi = (int)std::floor(t + 1e-4f);
            out[0] = std::max(0, lo);
            out[1] = std::min(n - 1, hi);
        };
        rng(p.x, g.origin.x, g.nx, io);
        rng(p.y, g.origin.y, g.ny, jo);
        rng(p.z, g.origin.z, g.nz, ko);
    };
    for (size_t e = 0; e < edges.size(); ++e)
    {
        const Vec3 &pa = nodes[edges[e].first];
        const Vec3 &pb = nodes[edges[e].second];
        int ia[2], ja[2], ka[2], ib[2], jb[2], kb[2];
        cellsTouching(pa, ia, ja, ka);
        cellsTouching(pb, ib, jb, kb);
        int i0 = std::max(ia[0], ib[0]), i1 = std::min(ia[1], ib[1]);
        int j0 = std::max(ja[0], jb[0]), j1 = std::min(ja[1], jb[1]);
        int k0 = std::max(ka[0], kb[0]), k1 = std::min(ka[1], kb[1]);
        bool pec = false;
        for (int k = k0; k <= k1 && !pec; ++k)
            for (int j = j0; j <= j1 && !pec; ++j)
                for (int i = i0; i <= i1 && !pec; ++i)
                    if (mat[g.cellIndex(i, j, k)] == MAT_PEC)
                        pec = true;
        edgePec[e] = pec ? 1 : 0;
    }

    // ---- outer boundary triangles (for the ABC) ----
    auto onBoundary = [&](int node, int axis, int side) -> bool
    {
        const Vec3 &p = nodes[node];
        float lim = (side == 0) ? g.origin[axis]
                                : g.origin[axis] + ((axis == 0) ? g.nx
                                  : (axis == 1) ? g.ny : g.nz) * g.dl;
        return std::fabs(p[axis] - lim) < 0.25f * g.dl;
    };
    static const int TF[4][3] = { {1,2,3}, {0,2,3}, {0,1,3}, {0,1,2} };
    for (const auto &tet : tets)
    {
        for (int f = 0; f < 4; ++f)
        {
            int q0 = tet.n[TF[f][0]], q1 = tet.n[TF[f][1]], q2 = tet.n[TF[f][2]];
            bool isB = false;
            for (int a = 0; a < 3 && !isB; ++a)
                for (int s = 0; s < 2 && !isB; ++s)
                    if (onBoundary(q0, a, s) && onBoundary(q1, a, s) &&
                        onBoundary(q2, a, s))
                        isB = true;
            if (!isB)
                continue;
            FemBFace bf;
            bf.n[0] = q0; bf.n[1] = q1; bf.n[2] = q2;
            static const int FE[3][2] = { {0,1}, {0,2}, {1,2} };
            for (int e = 0; e < 3; ++e)
            {
                int a = bf.n[FE[e][0]], b = bf.n[FE[e][1]];
                bf.e[e] = findEdge(a, b);
                bf.s[e] = (a < b) ? 1 : -1;
            }
            if (bf.e[0] >= 0 && bf.e[1] >= 0 && bf.e[2] >= 0)
                bfaces.push_back(bf);
        }
    }
    return !tets.empty();
}

//---------------------------------------------------------------------------
void FemMesh::cellPillarEdges(size_t cell, int polAxis,
                              std::vector<int> &out) const
{
    int i = (int)(cell % g.nx);
    int j = (int)((cell / g.nx) % g.ny);
    int k = (int)(cell / ((size_t)g.nx * g.ny));
    int t1 = (polAxis == 0) ? 1 : 0;
    int t2 = (polAxis == 2) ? 1 : 2;
    for (int d1 = 0; d1 < 2; ++d1)
        for (int d2 = 0; d2 < 2; ++d2)
        {
            int co0[3] = { i, j, k };
            co0[t1] += d1;
            co0[t2] += d2;
            int co1[3] = { co0[0], co0[1], co0[2] };
            co1[polAxis] += 1;
            int na = nodeId[nodeKey(co0[0], co0[1], co0[2])];
            int nb = nodeId[nodeKey(co1[0], co1[1], co1[2])];
            if (na < 0 || nb < 0)
                continue;
            int e = findEdge(na, nb);
            if (e >= 0)
                out.push_back(e);
        }
}

//---------------------------------------------------------------------------
void FemMesh::vizEdges(const std::vector<uint8_t> &mat,
                       std::vector<Vec3> &segments, size_t maxSegments) const
{
    segments.clear();
    // cells adjacent to PEC (the electrically interesting region)
    std::vector<uint8_t> show(mat.size(), 0);
    const int dirs[3] = { 1, g.nx, g.nx * g.ny };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                size_t c = g.cellIndex(i, j, k);
                if (mat[c] != MAT_PEC)
                    continue;
                const int co[3] = { i, j, k };
                const int nn[3] = { g.nx, g.ny, g.nz };
                for (int a = 0; a < 3; ++a)
                    for (int s = -1; s <= 1; s += 2)
                    {
                        int q = co[a] + s;
                        if (q < 0 || q >= nn[a])
                            continue;
                        show[c + (size_t)((long long)s * dirs[a])] = 1;
                    }
            }
    for (const auto &tet : tets)
    {
        if (!show[tet.cell] && tet.mat < MAT_DIEL0)
            continue;            // show PEC-adjacent + dielectric cells
        static const int EP[6][2] = { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
        for (int e = 0; e < 6; ++e)
        {
            segments.push_back(nodes[tet.n[EP[e][0]]]);
            segments.push_back(nodes[tet.n[EP[e][1]]]);
            if (segments.size() >= maxSegments * 2)
                return;
        }
    }
}
