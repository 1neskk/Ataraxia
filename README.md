# CUDARayTracer

## Get Started

### Linux

1. Clone the project repository

2. Build the project: `make build`

3. Run the resulting executable: `./build/CUDARayTracer`

### Windows

1. Clone the project repository

2. Build the project: `mkdir build && cmake -B build -G "Visual Studio 17 2022 (or whatever version you have)" `

3. Run the resulting solution file: `cd build && MSBuild CUDARayTracer.sln`

#### To Do
- [ ] Improve ray direction calculation (spheres on the sides are slightly stretched)
- [ ] Implement a way to change the camera position
