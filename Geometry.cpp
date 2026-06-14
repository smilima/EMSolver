//---------------------------------------------------------------------------
// Geometry.cpp - mesh utilities, STL import, voxelization
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop

#include "Geometry.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unordered_map>

#pragma package(smart_init)

//---------------------------------------------------------------------------
void ClusterDecimate(const TriMesh &in, float cell, TriMesh &out)
{
    out.clear();
    if (cell <= 0.0f || in.verts.empty())
    {
        out = in;
        return;
    }
    double inv = 1.0 / cell;
    auto key = [&](const Vec3 &p) -> long long
    {
        // pack three 21-bit cluster indices (offset to stay non-negative)
        long long x = (long long)std::llround(p.x * inv) + 1048576;
        long long y = (long long)std::llround(p.y * inv) + 1048576;
        long long z = (long long)std::llround(p.z * inv) + 1048576;
        x &= 0x1FFFFF; y &= 0x1FFFFF; z &= 0x1FFFFF;
        return (x << 42) | (y << 21) | z;
    };
    std::unordered_map<long long, int> cmap;
    std::vector<Vec3> accum;
    std::vector<int>  cnt;
    std::vector<int>  vmap(in.verts.size());
    for (size_t i = 0; i < in.verts.size(); ++i)
    {
        long long k = key(in.verts[i]);
        auto it = cmap.find(k);
        if (it == cmap.end())
        {
            int id = (int)accum.size();
            cmap[k] = id;
            vmap[i] = id;
            accum.push_back(in.verts[i]);
            cnt.push_back(1);
        }
        else
        {
            vmap[i] = it->second;
            accum[it->second] += in.verts[i];
            cnt[it->second]++;
        }
    }
    out.verts.resize(accum.size());
    for (size_t i = 0; i < accum.size(); ++i)
        out.verts[i] = accum[i] * (1.0f / cnt[i]);
    for (size_t t = 0; t + 2 < in.idx.size(); t += 3)
    {
        int a = vmap[in.idx[t]], b = vmap[in.idx[t + 1]], c = vmap[in.idx[t + 2]];
        if (a != b && b != c && a != c)
        {
            out.idx.push_back(a);
            out.idx.push_back(b);
            out.idx.push_back(c);
        }
    }
    out.computeNormals();
}

//---------------------------------------------------------------------------
// TriMesh
//---------------------------------------------------------------------------
void TriMesh::addTri(const Vec3 &a, const Vec3 &b, const Vec3 &c)
{
    int base = (int)verts.size();
    verts.push_back(a);
    verts.push_back(b);
    verts.push_back(c);
    idx.push_back(base);
    idx.push_back(base + 1);
    idx.push_back(base + 2);
}

//---------------------------------------------------------------------------
void TriMesh::addQuad(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d)
{
    addTri(a, b, c);
    addTri(a, c, d);
}

//---------------------------------------------------------------------------
void TriMesh::addPlate(const Vec3 &c, const Vec3 &u, const Vec3 &v)
{
    Vec3 hu = u * 0.5f, hv = v * 0.5f;
    addQuad(c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv);
}

//---------------------------------------------------------------------------
void TriMesh::addBox(const Vec3 &lo, const Vec3 &hi)
{
    Vec3 p000(lo.x, lo.y, lo.z), p100(hi.x, lo.y, lo.z);
    Vec3 p010(lo.x, hi.y, lo.z), p110(hi.x, hi.y, lo.z);
    Vec3 p001(lo.x, lo.y, hi.z), p101(hi.x, lo.y, hi.z);
    Vec3 p011(lo.x, hi.y, hi.z), p111(hi.x, hi.y, hi.z);
    addQuad(p000, p010, p110, p100); // -z
    addQuad(p001, p101, p111, p011); // +z
    addQuad(p000, p100, p101, p001); // -y
    addQuad(p010, p011, p111, p110); // +y
    addQuad(p000, p001, p011, p010); // -x
    addQuad(p100, p110, p111, p101); // +x
}

//---------------------------------------------------------------------------
void TriMesh::append(const TriMesh &m)
{
    int base = (int)verts.size();
    verts.insert(verts.end(), m.verts.begin(), m.verts.end());
    for (int i : m.idx)
        idx.push_back(base + i);
}

//---------------------------------------------------------------------------
void TriMesh::transform(const Vec3 &t)
{
    for (auto &v : verts)
        v += t;
}

//---------------------------------------------------------------------------
void TriMesh::computeNormals()
{
    normals.assign(verts.size(), Vec3(0, 0, 0));
    for (size_t t = 0; t + 2 < idx.size(); t += 3)
    {
        const Vec3 &a = verts[idx[t]], &b = verts[idx[t + 1]], &c = verts[idx[t + 2]];
        Vec3 n = (b - a).cross(c - a);
        normals[idx[t]]     += n;
        normals[idx[t + 1]] += n;
        normals[idx[t + 2]] += n;
    }
    for (auto &n : normals)
        n = n.normalized();
}

//---------------------------------------------------------------------------
Aabb TriMesh::bounds() const
{
    Aabb b;
    for (const auto &v : verts)
        b.grow(v);
    return b;
}

//---------------------------------------------------------------------------
Aabb SceneObject::worldBounds() const
{
    Aabb b;
    for (const auto &v : mesh.verts)
        b.grow(v + position);
    for (const auto &w : wires)
        for (const auto &p : w.pts)
        {
            b.grow(p + position + Vec3(w.radius, w.radius, w.radius));
            b.grow(p + position - Vec3(w.radius, w.radius, w.radius));
        }
    if (feed.enabled)
    {
        b.grow(feed.a + position);
        b.grow(feed.b + position);
    }
    return b;
}

//---------------------------------------------------------------------------
// STL import
//---------------------------------------------------------------------------
static bool LoadStlBinary(FILE *f, TriMesh &out, std::string &err)
{
    unsigned char header[80];
    if (fread(header, 1, 80, f) != 80) { err = "Truncated STL header"; return false; }
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { err = "Truncated STL count"; return false; }
    if (n == 0 || n > 50000000u) { err = "Unreasonable STL triangle count"; return false; }
    out.clear();
    out.verts.reserve((size_t)n * 3);
    out.idx.reserve((size_t)n * 3);
    for (uint32_t t = 0; t < n; ++t)
    {
        float rec[12];          // normal + 3 vertices
        uint16_t attr;
        if (fread(rec, 4, 12, f) != 12 || fread(&attr, 2, 1, f) != 1)
        {
            err = "Truncated STL facet data";
            return false;
        }
        out.addTri(Vec3(rec[3], rec[4],  rec[5]),
                   Vec3(rec[6], rec[7],  rec[8]),
                   Vec3(rec[9], rec[10], rec[11]));
    }
    return true;
}

//---------------------------------------------------------------------------
static bool LoadStlAscii(FILE *f, TriMesh &out, std::string &err)
{
    fseek(f, 0, SEEK_SET);
    out.clear();
    char line[512];
    Vec3 v[3];
    int nv = 0;
    while (fgets(line, sizeof(line), f))
    {
        const char *p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (strncmp(p, "vertex", 6) == 0)
        {
            float x, y, z;
            if (sscanf(p + 6, "%f %f %f", &x, &y, &z) == 3 && nv < 3)
                v[nv++] = Vec3(x, y, z);
        }
        else if (strncmp(p, "endfacet", 8) == 0)
        {
            if (nv == 3)
                out.addTri(v[0], v[1], v[2]);
            nv = 0;
        }
    }
    if (out.triCount() == 0) { err = "No triangles found in ASCII STL"; return false; }
    return true;
}

//---------------------------------------------------------------------------
bool LoadStl(const std::wstring &path, TriMesh &out, std::string &err)
{
    FILE *f = _wfopen(path.c_str(), L"rb");
    if (!f) { err = "Cannot open file"; return false; }

    // Detect format: ASCII files start with "solid" AND parse as text; but some
    // binary files also start with "solid", so verify with the size formula.
    char head[6] = {0};
    fread(head, 1, 5, f);
    bool ok;
    bool looksAscii = (strncmp(head, "solid", 5) == 0);
    if (looksAscii)
    {
        // Check binary size consistency to catch "solid"-prefixed binaries
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 80, SEEK_SET);
        uint32_t n = 0;
        if (fread(&n, 4, 1, f) == 1 && size == 84L + 50L * (long)n && n > 0)
            looksAscii = false;
    }
    if (looksAscii)
        ok = LoadStlAscii(f, out, err);
    else
    {
        fseek(f, 0, SEEK_SET);
        ok = LoadStlBinary(f, out, err);
    }
    fclose(f);
    if (ok)
        out.computeNormals();
    return ok;
}

//---------------------------------------------------------------------------
// Triangle/box overlap (Akenine-Moller separating axis test)
//---------------------------------------------------------------------------
static inline bool AxisTest(float a, float b, float fa, float fb,
                            float v0a, float v0b, float v2a, float v2b,
                            float ha, float hb)
{
    float p0 = a * v0a - b * v0b;
    float p2 = a * v2a - b * v2b;
    float mn = p0 < p2 ? p0 : p2, mx = p0 < p2 ? p2 : p0;
    float rad = fa * ha + fb * hb;
    return !(mn > rad || mx < -rad);
}

//---------------------------------------------------------------------------
bool TriBoxOverlap(const Vec3 &c, const Vec3 &h,
                   const Vec3 &tv0, const Vec3 &tv1, const Vec3 &tv2)
{
    Vec3 v0 = tv0 - c, v1 = tv1 - c, v2 = tv2 - c;
    Vec3 e0 = v1 - v0, e1 = v2 - v1, e2 = v0 - v2;

    // 9 cross-product axes
    float fex = std::fabs(e0.x), fey = std::fabs(e0.y), fez = std::fabs(e0.z);
    if (!AxisTest(e0.z, e0.y, fez, fey, v0.y, v0.z, v2.y, v2.z, h.y, h.z)) return false;
    if (!AxisTest(e0.z, e0.x, fez, fex, v0.x, v0.z, v2.x, v2.z, h.x, h.z)) return false;
    if (!AxisTest(e0.y, e0.x, fey, fex, v1.x, v1.y, v2.x, v2.y, h.x, h.y)) return false;

    fex = std::fabs(e1.x); fey = std::fabs(e1.y); fez = std::fabs(e1.z);
    if (!AxisTest(e1.z, e1.y, fez, fey, v0.y, v0.z, v2.y, v2.z, h.y, h.z)) return false;
    if (!AxisTest(e1.z, e1.x, fez, fex, v0.x, v0.z, v2.x, v2.z, h.x, h.z)) return false;
    if (!AxisTest(e1.y, e1.x, fey, fex, v0.x, v0.y, v1.x, v1.y, h.x, h.y)) return false;

    fex = std::fabs(e2.x); fey = std::fabs(e2.y); fez = std::fabs(e2.z);
    if (!AxisTest(e2.z, e2.y, fez, fey, v0.y, v0.z, v1.y, v1.z, h.y, h.z)) return false;
    if (!AxisTest(e2.z, e2.x, fez, fex, v0.x, v0.z, v1.x, v1.z, h.x, h.z)) return false;
    if (!AxisTest(e2.y, e2.x, fey, fex, v1.x, v1.y, v2.x, v2.y, h.x, h.y)) return false;

    // box face axes
    auto minMax3 = [](float a, float b, float cc, float &mn, float &mx)
    {
        mn = mx = a;
        if (b < mn) mn = b; if (b > mx) mx = b;
        if (cc < mn) mn = cc; if (cc > mx) mx = cc;
    };
    float mn, mx;
    minMax3(v0.x, v1.x, v2.x, mn, mx); if (mn > h.x || mx < -h.x) return false;
    minMax3(v0.y, v1.y, v2.y, mn, mx); if (mn > h.y || mx < -h.y) return false;
    minMax3(v0.z, v1.z, v2.z, mn, mx); if (mn > h.z || mx < -h.z) return false;

    // triangle plane vs box
    Vec3 n = e0.cross(e1);
    Vec3 vmnC, vmxC;
    for (int q = 0; q < 3; ++q)
    {
        float v0q = v0[q], hq = h[q];
        if (n[q] > 0.0f) { vmnC.set(q, -hq - v0q); vmxC.set(q,  hq - v0q); }
        else             { vmnC.set(q,  hq - v0q); vmxC.set(q, -hq - v0q); }
    }
    if (n.dot(vmnC) > 0.0f) return false;
    if (n.dot(vmxC) >= 0.0f) return true;
    return false;
}

//---------------------------------------------------------------------------
// Voxelizers
//---------------------------------------------------------------------------
void VoxelizeMesh(const TriMesh &mesh, const Vec3 &offset,
                  const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                  uint8_t matId)
{
    const Vec3 half(g.dl * 0.5f, g.dl * 0.5f, g.dl * 0.5f);
    for (int t = 0; t < mesh.triCount(); ++t)
    {
        Vec3 a = mesh.verts[mesh.idx[t * 3]]     + offset;
        Vec3 b = mesh.verts[mesh.idx[t * 3 + 1]] + offset;
        Vec3 c = mesh.verts[mesh.idx[t * 3 + 2]] + offset;
        Aabb tb;
        tb.grow(a); tb.grow(b); tb.grow(c);
        int i0, j0, k0, i1, j1, k1;
        g.cellOf(tb.lo, i0, j0, k0);
        g.cellOf(tb.hi, i1, j1, k1);
        i0 = std::max(i0, 0); j0 = std::max(j0, 0); k0 = std::max(k0, 0);
        i1 = std::min(i1, g.nx - 1); j1 = std::min(j1, g.ny - 1); k1 = std::min(k1, g.nz - 1);
        for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i)
                {
                    size_t ci = g.cellIndex(i, j, k);
                    if (mat[ci] == matId)
                        continue;
                    if (TriBoxOverlap(g.cellCenter(i, j, k), half, a, b, c))
                        mat[ci] = matId;
                }
    }
}

//---------------------------------------------------------------------------
// PEC surface faces (shared by TLM and FEM solvers)
//---------------------------------------------------------------------------
void BuildPecSurfaceFaces(const VoxelGridSpec &g,
                          const std::vector<uint8_t> &mat, uint8_t matPec,
                          std::vector<SurfaceFace> &out)
{
    out.clear();
    const int dirs[3] = { 1, g.nx, g.nx * g.ny };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                size_t c = g.cellIndex(i, j, k);
                if (mat[c] != matPec)
                    continue;
                const int co[3] = { i, j, k };
                const int nn[3] = { g.nx, g.ny, g.nz };
                for (int a = 0; a < 3; ++a)
                    for (int s = -1; s <= 1; s += 2)
                    {
                        int q = co[a] + s;
                        if (q < 0 || q >= nn[a])
                            continue;
                        size_t nb = c + (size_t)((long long)s * dirs[a]);
                        if (mat[nb] == matPec)
                            continue;
                        SurfaceFace f;
                        f.airCell = (int)nb;
                        f.axis    = (int8_t)a;
                        f.sign    = (int8_t)s;
                        Vec3 cc   = g.cellCenter(i, j, k);
                        cc.set(a, cc[a] + s * g.dl * 0.5f);
                        f.center  = cc;
                        out.push_back(f);
                    }
            }
}

//---------------------------------------------------------------------------
// Voxel-block wireframe for grid preview (TLM/FDTD)
//---------------------------------------------------------------------------
void BuildVoxelOutline(const VoxelGridSpec &g,
                       const std::vector<uint8_t> &mat,
                       std::vector<Vec3> &segments, size_t maxSegments)
{
    segments.clear();
    // dedup via packed node-pair keys (node ids fit in 21 bits each)
    std::vector<uint64_t> seen;
    seen.reserve(1 << 16);
    auto nodeKey = [&](int i, int j, int k) -> uint64_t
    {
        return ((uint64_t)k * (g.ny + 1) + j) * (g.nx + 1) + i;
    };
    // simple open-addressing set
    size_t cap = 1;
    std::vector<uint64_t> table;
    auto rehash = [&](size_t newCap)
    {
        std::vector<uint64_t> old = std::move(table);
        cap = newCap;
        table.assign(cap, ~0ull);
        for (uint64_t v : old)
            if (v != ~0ull)
            {
                size_t h = (size_t)((v * 0x9E3779B97F4A7C15ull) & (cap - 1));
                while (table[h] != ~0ull)
                    h = (h + 1) & (cap - 1);
                table[h] = v;
            }
    };
    rehash(1 << 17);
    size_t used = 0;
    auto insertOnce = [&](uint64_t key) -> bool
    {
        size_t h = (size_t)((key * 0x9E3779B97F4A7C15ull) & (cap - 1));
        while (table[h] != ~0ull)
        {
            if (table[h] == key)
                return false;
            h = (h + 1) & (cap - 1);
        }
        table[h] = key;
        if (++used * 2 > cap)
            rehash(cap * 2);
        return true;
    };

    auto nodePos = [&](int i, int j, int k)
    {
        return Vec3(g.origin.x + i * g.dl, g.origin.y + j * g.dl,
                    g.origin.z + k * g.dl);
    };
    // 12 cube edges as corner-offset pairs
    static const int CE[12][2][3] = {
        {{0,0,0},{1,0,0}}, {{0,1,0},{1,1,0}}, {{0,0,1},{1,0,1}}, {{0,1,1},{1,1,1}},
        {{0,0,0},{0,1,0}}, {{1,0,0},{1,1,0}}, {{0,0,1},{0,1,1}}, {{1,0,1},{1,1,1}},
        {{0,0,0},{0,0,1}}, {{1,0,0},{1,0,1}}, {{0,1,0},{0,1,1}}, {{1,1,0},{1,1,1}} };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                if (mat[g.cellIndex(i, j, k)] == 0 /* MAT_AIR */)
                    continue;
                for (int e = 0; e < 12; ++e)
                {
                    int a[3] = { i + CE[e][0][0], j + CE[e][0][1],
                                 k + CE[e][0][2] };
                    int b[3] = { i + CE[e][1][0], j + CE[e][1][1],
                                 k + CE[e][1][2] };
                    uint64_t ka = nodeKey(a[0], a[1], a[2]);
                    uint64_t kb = nodeKey(b[0], b[1], b[2]);
                    uint64_t key = (std::min(ka, kb) << 28) | std::max(ka, kb);
                    if (!insertOnce(key))
                        continue;
                    segments.push_back(nodePos(a[0], a[1], a[2]));
                    segments.push_back(nodePos(b[0], b[1], b[2]));
                    if (segments.size() >= maxSegments * 2)
                        return;
                }
            }
}

//---------------------------------------------------------------------------
// Moller-Trumbore ray/triangle; ray is +x from origin o. Returns hit x.
static bool RayXTri(const Vec3 &o, const Vec3 &a, const Vec3 &b, const Vec3 &c,
                    float &hitX)
{
    const Vec3 dir(1, 0, 0);
    Vec3 e1 = b - a, e2 = c - a;
    Vec3 p = dir.cross(e2);
    float det = e1.dot(p);
    if (std::fabs(det) < 1e-12f)
        return false;
    float inv = 1.0f / det;
    Vec3 tv = o - a;
    float u = tv.dot(p) * inv;
    if (u < 0.0f || u > 1.0f)
        return false;
    Vec3 q = tv.cross(e1);
    float v = dir.dot(q) * inv;
    if (v < 0.0f || u + v > 1.0f)
        return false;
    float t = e2.dot(q) * inv;
    if (t <= 0.0f)
        return false;
    hitX = o.x + t;
    return true;
}

//---------------------------------------------------------------------------
void VoxelizeMeshSolid(const TriMesh &mesh, const Vec3 &offset,
                       const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                       uint8_t matId)
{
    // surface pass first (covers parity-fragile boundary cells)
    VoxelizeMesh(mesh, offset, g, mat, matId);

    Aabb mb;
    for (const auto &v : mesh.verts)
        mb.grow(v + offset);
    if (!mb.valid())
        return;
    int i0, j0, k0, i1, j1, k1;
    g.cellOf(mb.lo, i0, j0, k0);
    g.cellOf(mb.hi, i1, j1, k1);
    i0 = std::max(i0, 0); j0 = std::max(j0, 0); k0 = std::max(k0, 0);
    i1 = std::min(i1, g.nx - 1); j1 = std::min(j1, g.ny - 1); k1 = std::min(k1, g.nz - 1);

    std::vector<float> hits;
    for (int k = k0; k <= k1; ++k)
        for (int j = j0; j <= j1; ++j)
        {
            Vec3 o = g.cellCenter(i0, j, k);
            o.x = mb.lo.x - g.dl;        // start outside the mesh
            hits.clear();
            for (int t = 0; t < mesh.triCount(); ++t)
            {
                float hx;
                if (RayXTri(o, mesh.verts[mesh.idx[t*3]] + offset,
                               mesh.verts[mesh.idx[t*3+1]] + offset,
                               mesh.verts[mesh.idx[t*3+2]] + offset, hx))
                    hits.push_back(hx);
            }
            if (hits.size() < 2)
                continue;
            std::sort(hits.begin(), hits.end());
            // walk in/out pairs, drop duplicates from shared edges
            for (size_t h = 0; h + 1 < hits.size(); h += 2)
            {
                float xa = hits[h], xb = hits[h + 1];
                int ia, ib, dum1, dum2;
                Vec3 pa(xa, 0, 0), pb(xb, 0, 0);
                g.cellOf(pa, ia, dum1, dum2);
                g.cellOf(pb, ib, dum1, dum2);
                ia = std::max(ia, i0);
                ib = std::min(ib, i1);
                for (int i = ia; i <= ib; ++i)
                {
                    float cx = g.origin.x + (i + 0.5f) * g.dl;
                    if (cx >= xa && cx <= xb)
                        mat[g.cellIndex(i, j, k)] = matId;
                }
            }
        }
}

//---------------------------------------------------------------------------
void VoxelizeWire(const Wire &w, const Vec3 &offset,
                  const VoxelGridSpec &g, std::vector<uint8_t> &mat,
                  uint8_t matId)
{
    auto stamp = [&](const Vec3 &p)
    {
        int rc = (int)std::ceil(w.radius / g.dl - 0.5f); // extra cells around center
        if (rc < 0) rc = 0;
        int ic, jc, kc;
        g.cellOf(p, ic, jc, kc);
        for (int k = kc - rc; k <= kc + rc; ++k)
            for (int j = jc - rc; j <= jc + rc; ++j)
                for (int i = ic - rc; i <= ic + rc; ++i)
                {
                    if (!g.inGrid(i, j, k))
                        continue;
                    if (rc > 0)
                    {
                        Vec3 d = g.cellCenter(i, j, k) - p;
                        if (d.length() > w.radius + g.dl * 0.5f)
                            continue;
                    }
                    mat[g.cellIndex(i, j, k)] = matId;
                }
    };

    for (size_t s = 0; s + 1 < w.pts.size(); ++s)
    {
        Vec3 a = w.pts[s] + offset, b = w.pts[s + 1] + offset;
        float len = (b - a).length();
        int steps = std::max(1, (int)std::ceil(len / (g.dl * 0.25f)));
        for (int i = 0; i <= steps; ++i)
            stamp(a + (b - a) * ((float)i / steps));
    }
}
