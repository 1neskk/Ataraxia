#pragma once

#include <cuda_runtime.h>
#include <iostream>

template <class T>
class vec2
{
public:
    union
    {
        struct { T x, y; };
        struct { T u, v; };
        struct { T V[2]; };
    };

    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(T x, T y) : x(x), y(y) {}
    __host__ __device__ vec2(const vec2& v) : x(v.x), y(v.y) {}

    // operators
    __host__ __device__ vec2& operator=(const vec2& v) { x = v.x; y = v.y; return *this; }

    __host__ __device__ vec2& operator+=(const vec2& v) { x += v.x; y += v.y; return *this; }
    __host__ __device__ vec2& operator-=(const vec2& v) { x -= v.x; y -= v.y; return *this; }
    __host__ __device__ vec2& operator*=(T s) { x *= s; y *= s; return *this; }
    __host__ __device__ vec2& operator/=(T s) { x /= s; y /= s; return *this; }

    __host__ __device__ vec2 operator+(const vec2& v) const { return vec2(x + v.x, y + v.y); }
    __host__ __device__ vec2 operator-(const vec2& v) const { return vec2(x - v.x, y - v.y); }
    __host__ __device__ vec2 operator*(T s) const { return vec2(x * s, y * s); }
    __host__ __device__ vec2 operator/(T s) const { return vec2(x / s, y / s); }

    __host__ __device__ friend vec2 operator*(T s, const vec2& v) { return vec2(s * v.x, s * v.y); }
    __host__ __device__ friend vec2 operator/(T s, const vec2& v) { return vec2(s / v.x, s / v.y); }

    __host__ __device__ vec2& operator++() { x++; y++; return *this; }
    __host__ __device__ vec2& operator--() { x--; y--; return *this; }
    __host__ __device__ vec2 operator++(int) { vec2 v(*this); x++; y++; return v; }
    __host__ __device__ vec2 operator--(int) { vec2 v(*this); x--; y--; return v; }

    __host__ __device__ vec2 operator-() const { return vec2(-x, -y); }

    __host__ __device__ bool operator==(const vec2& v) const { return x == v.x && y == v.y; }
    __host__ __device__ bool operator!=(const vec2& v) const { return x != v.x || y != v.y; }

    __host__ __device__ T dot(const vec2& v) const { return x * v.x + y * v.y; }
    __host__ __device__ T length() const { return sqrt(x * x + y * y); }
    __host__ __device__ vec2 normalize() const { return *this / length(); }

    friend std::ostream& operator<<(std::ostream& os, const vec2& v)
    {
        os << "(" << v.x << ", " << v.y << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, vec2& v)
    {
        is >> v.x >> v.y;
        return is;
    }

    T& operator[](int i) { return V[i]; }
    const T& operator[](int i) const { return V[i]; }
};
typedef vec2<float> Vec2;
