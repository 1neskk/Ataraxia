#include "Utils.h"

json Utils::serializeScene(const Scene& scene)
{
    json j;

    scene.rootNode->updateGlobalTransform();
    j["camera"] =
    {
        {"position", {scene.camera.getPosition().x, scene.camera.getPosition().y, scene.camera.getPosition().z}},
        {"direction", {scene.camera.getDirection().x, scene.camera.getDirection().y, scene.camera.getDirection().z}},
        {"fov", scene.camera.getFov()}
    };

    if (scene.rootNode)
    {
        j["sceneGraph"] = serializeSceneNode(scene.rootNode);
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

json Utils::serializeSceneNode(const std::shared_ptr<SceneNode>& node)
{
    json j;

    j["name"] = node->getName();
    j["transformation"] =
    {
        {"position", {node->getPosition().x, node->getPosition().y, node->getPosition().z}},
        {"rotation", {node->getRotation().x, node->getRotation().y, node->getRotation().z, node->getRotation().w}},
        {"scale", {node->getScale().x, node->getScale().y, node->getScale().z}}
    };

    for (const auto& sphere : node->getSpheres())
    {
        j["spheres"].push_back(
        {
            {"center", {sphere.center.x, sphere.center.y, sphere.center.z}},
            {"radius", sphere.radius},
            {"materialIndex", sphere.id}
        });
    }

    if (!node->getChildren().empty())
    {
        for (const auto& child : node->getChildren())
        {
            j["children"].push_back(serializeSceneNode(child));
        }
    }

    return j;
}

Scene Utils::deserializeScene(const json& j)
{
    Scene scene;

    scene.camera.setPosition(glm::vec3(j["camera"]["position"][0], j["camera"]["position"][1], j["camera"]["position"][2]));
    scene.camera.setDirection(glm::vec3(j["camera"]["direction"][0], j["camera"]["direction"][1], j["camera"]["direction"][2]));
    scene.camera.setFov(j["camera"]["fov"]);

    if (j.contains("sceneGraph"))
    {
        deserializeSceneNode(j["sceneGraph"], scene.rootNode);
		scene.rootNode->updateGlobalTransform();
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

void Utils::deserializeSceneNode(const json& j, std::shared_ptr<SceneNode>& node)
{
    if (!node)
        node = std::make_shared<SceneNode>(j["name"].get<std::string>());

    glm::vec3 position(j["transformation"]["position"][0], j["transformation"]["position"][1], j["transformation"]["position"][2]);
    glm::quat rotation(j["transformation"]["rotation"][3], j["transformation"]["rotation"][0], j["transformation"]["rotation"][1], j["transformation"]["rotation"][2]);
    glm::vec3 scale(j["transformation"]["scale"][0], j["transformation"]["scale"][1], j["transformation"]["scale"][2]);

    node->setPosition(position);
    node->setRotation(rotation);
    node->setScale(scale);

    if (j.contains("spheres"))
    {
        for (const auto& sphere : j["spheres"])
        {
            Sphere s;
            s.center = glm::vec3(sphere["center"][0], sphere["center"][1], sphere["center"][2]);
            s.radius = sphere["radius"];
            s.id = sphere["materialIndex"];
            node->addSphere(s);
        }
    }

	if (j.contains("children"))
	{
		for (const auto& child : j["children"])
		{
			std::shared_ptr<SceneNode> childNode;
			deserializeSceneNode(child, childNode);
			node->addChild(childNode);
		}
	}
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
