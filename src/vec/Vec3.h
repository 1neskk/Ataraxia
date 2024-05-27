#pragma once

#include <cuda_runtime.h>
#include <iostream>

template <class T>
class vec3
{
public:
    union
    {
        struct { T x, y, z; };
        struct { T r, g, b; };
        struct { T V[3]; };
    };

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(T x, T y, T z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3(const vec3& v) : x(v.x), y(v.y), z(v.z) {}

    // operators
    __host__ __device__ vec3& operator=(const vec3& v) { x = v.x; y = v.y; z = v.z; return *this; }

    __host__ __device__ vec3& operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    __host__ __device__ vec3& operator-=(const vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    __host__ __device__ vec3& operator*=(T s) { x *= s; y *= s; z *= s; return *this; }
    __host__ __device__ vec3& operator/=(T s) { x /= s; y /= s; z /= s; return *this; }

    __host__ __device__ vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ vec3 operator*(T s) const { return vec3(x * s, y * s, z * s); }
    __host__ __device__ vec3 operator/(T s) const { return vec3(x / s, y / s, z / s); }

    __host__ __device__ friend vec3 operator*(T s, const vec3& v) { return vec3(s * v.x, s * v.y, s * v.z); }
    __host__ __device__ friend vec3 operator/(T s, const vec3& v) { return vec3(s / v.x, s / v.y, s / v.z); }

    __host__ __device__ vec3& operator++() { x++; y++; z++; return *this; }
    __host__ __device__ vec3& operator--() { x--; y--; z--; return *this; }
    __host__ __device__ vec3 operator++(int) { vec3 v(*this); x++; y++; z++; return v; }
    __host__ __device__ vec3 operator--(int) { vec3 v(*this); x--; y--; z--; return v; }

    __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }

    __host__ __device__ bool operator==(const vec3& v) const { return x == v.x && y == v.y && z == v.z; }
    __host__ __device__ bool operator!=(const vec3& v) const { return x != v.x || y != v.y || z != v.z; }

    __host__ __device__ T dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    __host__ __device__ vec3 cross(const vec3& v) const { return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
    __host__ __device__ T length() const { return sqrt(x * x + y * y + z * z); }
    __host__ __device__ vec3 normalize() const { return *this / length(); }

    friend std::ostream& operator<<(std::ostream& os, const vec3& v)
    {
        os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, vec3& v)
    {
        is >> v.x >> v.y >> v.z;
        return is;
    }

    T& operator[](int i) { return V[i]; }
    const T& operator[](int i) const { return V[i]; }
};
typedef vec3<float> Vec3;