# CUDARayTracer

![image](https://github.com/1neskk/CUDARayTracer/assets/113075816/848d0f6f-6f14-4836-822f-560e018a7e3f)

This project uses thirdparty libraries such as [GLFW](https://github.com/glfw/glfw), [GLM](https://github.com/g-truc/glm), [Dear ImGui](https://github.com/ocornut/imgui) and [stb](https://github.com/nothings/stb)

**Officially supports Windows 10/11 with Visual Studio 2022 and Linux (tested on the LTS kernel with Wayland)**

## Requirements
- [Vulkan SDK](https://vulkan.lunarg.com/)
- NVIDIA GPU
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

## Get Started

### Linux

1. Clone recursively: `git clone --recursive https://github.com/1neskk/CUDARayTracer`

2. Build the project: `make build`

3. Run the resulting executable: `./build/CUDARayTracer`

### Windows

1. Clone recursively: `git clone --recursive https://github.com/1neskk/CUDARayTracer`

2. Build the project: `mkdir build && cmake -B build -G "Visual Studio 17 2022" -A x64 `

3. Run the resulting solution file: `cd build && msbuild CUDARayTracer.sln /p:Configuration=Release /p:Platform=x64`

#### To Do
- [ ] Material system
- [ ] Texture mapping
- [ ] HDR
- [ ] Shadows
