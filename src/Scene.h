#pragma once

#include "vec/Vec3.h"
#include <vector>

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct Sphere
{
    Vec3 center;
    float radius;
    int id = 0;
};

struct Scene
{
    Sphere* spheres;
    size_t numSpheres;

    Scene() : spheres(nullptr), numSpheres(0) {}

    void setSpheres(const std::vector<Sphere>& sphereVec)
    {
        numSpheres = sphereVec.size();
        cudaMalloc(&spheres, numSpheres * sizeof(Sphere));
        cudaMemcpy(spheres, sphereVec.data(), numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    }

    ~Scene()
    {
        if (spheres)
            cudaFree(spheres);
    }

};
