#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../thirdparty/stb/stb_image_write.h"

struct Vec3
{
	float x, y, z;
	__host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	__host__ __device__ Vec3(const float x, const float y, const float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return { x + v.x, y + v.y, z + v.z }; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return { x - v.x, y - v.y, z - v.z }; }
    __host__ __device__ Vec3 operator*(const float s) const { return { x * s, y * s, z * s }; }
    __host__ __device__ Vec3 operator/(const float s) const { return { x / s, y / s, z / s }; }

    // scalar
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return { x * v.x, y * v.y, z * v.z }; }
    __host__ __device__ Vec3 operator/(const Vec3& v) const { return { x / v.x, y / v.y, z / v.z }; }

	__host__ __device__ float length() const { return sqrtf(x * x + y * y + z * z); }
	__host__ __device__ Vec3 normalize() const { return *this / length(); }
	__host__ __device__ static float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
	
	__host__ __device__ static Vec3 cross(const Vec3& a, const Vec3& b)
    {
        return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
    }

};

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct Sphere
{
    Vec3 center;
    float radius;
};

