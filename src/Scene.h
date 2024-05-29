#pragma once

#include <glm/glm.hpp>
#include <vector>

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Sphere
{
    glm::vec3 center;
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
