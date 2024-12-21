#include "Utils.h"

json Utils::serializeScene(const Scene& scene)
{
    json j;

    j["camera"] =
    {
        {"position", {scene.camera.getPosition().x, scene.camera.getPosition().y, scene.camera.getPosition().z}},
        {"direction", {scene.camera.getDirection().x, scene.camera.getDirection().y, scene.camera.getDirection().z}},
        {"fov", scene.camera.getFov()}
    };

    for (const auto& sphere : scene.spheres)
    {
        j["spheres"].push_back(
        {
            {"center", {sphere.center.x, sphere.center.y, sphere.center.z}},
            {"radius", sphere.radius},
            {"materialIndex", sphere.id}
        });
    }

    for (const auto& material : scene.materials)
    {
        j["materials"].push_back(
        {
            {"albedo", {material.albedo.r, material.albedo.g, material.albedo.b}},
            {"roughness", material.roughness},
            {"metallic", material.metallic},
            {"F0", {material.F0.r, material.F0.g, material.F0.b}},
            {"emissionIntensity", material.emissionIntensity},
            {"emissionColor", {material.emissionColor.r, material.emissionColor.g, material.emissionColor.b}}
        });
    }

    for (const auto& light : scene.lights)
    {
        j["lights"].push_back(
        {
            {"position", {light.position.x, light.position.y, light.position.z}},
            {"intensity", light.intensity},
            {"color", {light.color.r, light.color.g, light.color.b}}
        });
    }

    j["settings"] =
    {
        {"maxBounces", scene.settings.maxBounces},
        {"skyLight", scene.settings.skyLight},
        {"accumulation", scene.settings.accumulation},
    };

    return j;
}

void Utils::exportScene(const Scene& scene, const std::string& filename)
{
    const json j = serializeScene(scene);
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << j.dump(4); // indent with 4 spaces
        file.close();
    }
}

Scene Utils::deserializeScene(const json& j)
{
    Scene scene;

    scene.camera.setPosition(glm::vec3(j["camera"]["position"][0], j["camera"]["position"][1], j["camera"]["position"][2]));
    scene.camera.setDirection(glm::vec3(j["camera"]["direction"][0], j["camera"]["direction"][1], j["camera"]["direction"][2]));
    scene.camera.setFov(j["camera"]["fov"]);

    for (const auto& sphere : j["spheres"])
    {
        Sphere s;
        s.center = glm::vec3(sphere["center"][0], sphere["center"][1], sphere["center"][2]);
        s.radius = sphere["radius"];
        s.id = sphere["materialIndex"];
        scene.spheres.push_back(s);
    }

    for (const auto& material : j["materials"])
    {
        Material m;
        m.albedo = glm::vec3(material["albedo"][0], material["albedo"][1], material["albedo"][2]);
        m.roughness = material["roughness"];
        m.metallic = material["metallic"];
        m.F0 = glm::vec3(material["F0"][0], material["F0"][1], material["F0"][2]);
        m.emissionIntensity = material["emissionIntensity"];
        m.emissionColor = glm::vec3(material["emissionColor"][0], material["emissionColor"][1], material["emissionColor"][2]);
        scene.materials.push_back(m);
    }

    for (const auto& light : j["lights"])
    {
        Light l;
        l.position = glm::vec3(light["position"][0], light["position"][1], light["position"][2]);
        l.intensity = light["intensity"];
        l.color = glm::vec3(light["color"][0], light["color"][1], light["color"][2]);
        scene.lights.push_back(l);
    }

    scene.settings.maxBounces = j["settings"]["maxBounces"];
    scene.settings.skyLight = j["settings"]["skyLight"];
    scene.settings.accumulation = j["settings"]["accumulation"];

    return scene;
}

Scene Utils::importScene(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return Scene();

    json j;
    file >> j;
    file.close();

    Scene scene = deserializeScene(j);
    return scene;
}
