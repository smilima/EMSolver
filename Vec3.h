//---------------------------------------------------------------------------
// Vec3.h - minimal 3D vector math for the RF simulator
//---------------------------------------------------------------------------
#ifndef Vec3H
#define Vec3H

#include <cmath>

struct Vec3
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float ax, float ay, float az) : x(ax), y(ay), z(az) {}

    Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s)       const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s)       const { return Vec3(x / s, y / s, z / s); }
    Vec3 operator-()              const { return Vec3(-x, -y, -z); }
    Vec3 &operator+=(const Vec3 &v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3 &operator-=(const Vec3 &v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3 &operator*=(float s)       { x *= s; y *= s; z *= s; return *this; }

    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
    void  set(int i, float v)     { if (i == 0) x = v; else if (i == 1) y = v; else z = v; }

    float dot(const Vec3 &v)   const { return x * v.x + y * v.y + z * v.z; }
    Vec3  cross(const Vec3 &v) const
    {
        return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }
    float length()  const { return std::sqrt(x * x + y * y + z * z); }
    float length2() const { return x * x + y * y + z * z; }
    Vec3  normalized() const
    {
        float l = length();
        return l > 1e-20f ? (*this) / l : Vec3(0, 0, 0);
    }
};

inline Vec3 operator*(float s, const Vec3 &v) { return v * s; }

inline Vec3 vmin(const Vec3 &a, const Vec3 &b)
{
    return Vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}
inline Vec3 vmax(const Vec3 &a, const Vec3 &b)
{
    return Vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}

// Axis-aligned bounding box
struct Aabb
{
    Vec3 lo{ 1e30f,  1e30f,  1e30f};
    Vec3 hi{-1e30f, -1e30f, -1e30f};
    void grow(const Vec3 &p) { lo = vmin(lo, p); hi = vmax(hi, p); }
    void grow(const Aabb &b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
    bool valid() const { return lo.x <= hi.x; }
    Vec3 center() const { return (lo + hi) * 0.5f; }
    Vec3 size()   const { return hi - lo; }
};

#endif
