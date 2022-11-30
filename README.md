# CSE167 Final Project Write Up

#### **This is a personal project implementing a ray tracer**

## Brief Introduction
This project implements a ray tracer from scratch. The reason why I gave up using the starter code in Homework3 lies on that OpenGL is basically a rasterizer and not so good for building ray tracer (although it can). Also, I want to implement some optimized features to speed up rendering (for example, building BVH). Therefore, I rewrite this renderer. 

This renderer aims to have the following features:
- Bounding Volume Hierachy based on SAH
- Multiple Samples Anti-aliasing
- light 
- material: diffuse, metal, ...
- .......
- Parallel rendering with CPU (OpenMP)
- stack (?)

## Code Explanatoin
This part is the code document for describing how each module works and relative maths and algorithms.

### Ray
#### Define ray
Mathematically, a ray is 
$$R(t) = A + t \vec{d} (t > 0)$$
A is the origin, t is the length of direction vector.
Therefore, define a ```ray``` class like this:
```cpp
struct Ray{
    //Destination = origin + t*direction
    Vector3f origin;
    Vector3f direction, direction_inv;
    double t;
};
```
Notice that there is also a ```direction_inv``` which is (1/directioin.x, 1/directioin.y, 1/direction.z). The reason for defining this is to speed up the AABB hit method afterwards.

### Scene
Define the scene of the world.
- Frame(image) paramters: width / height
- Camera paramters: imageAspectRatio / fovy / eyePos / target / vup
- 
#### Casting ray to viewport
The viewport (aka screen) coordinate ranges from x[-1, 1] and y[-1, 1]. 
Given the imageAspectRatio and fovy, then 
$$viewportHeight = tan(\frac{fovy}{2}) * 2$$
$$viewportWidth = imageAspectRatio * viewportHeight$$
For pixel in (i, j), the center coordinate is 
$$x = 2 *\frac{i + 0.5}{2} - 1, y = 1 - \frac{i + 0.5}{2}$$

Given the eyePos, target, and vup vector, calculate 
$$w = normalize(eyePos - target), u = normalize(cross(vup, w)), v = normalize(cross(w, u))$$

then the ray direction is 
$$normalize(x * viewportWidth * u + y * viewportHeight * v - w)$$

#### Mutiple samples Anti-aliasing
Sample the pixel multiple times and average the result by the end. For each ray in a single pixel, send it to different points near the center by adding a random float to the center coordinate. Now the ray's target coordinate is:
$$x = 2 *\frac{i + 0.5 + randomFloat }{2} - 1, y = 1 - \frac{i + 0.5 + randomFloat}{2}$$

#### Build Bounding Volume Hierachy
If calculate the intersection for every single object, the time cost will be linear to the number of objects, which is the main time-bottleneck. Therefore, before doing the ray intersection part, build a BVH to speed up the procedure. 

1.  
### Renderer

### Image
Write the frame into a ppm file. Before writing, use gamma correction for accurate color intensity.
$$clamp(0, 1, color)^{0.6}$$
### Vector3f
Define ```Vector3f``` class for 3D vector operations

### Global
Define some useful global util functions (clamp, deg2Rad, etc..)

...TODO
## Build and Run
```sh
cd src
mkdir build
cd build
cmake ..
make
./RayTracing
```

## CUDA version
I don't want to lose the feature of parallel rendering under GPU. So I also try to implement it on CUDA. Due to the limitation of time, I only implement some basic functions (not a real RT like the above CPU-version)

### Code implementation
todo
### Build and Run
The code is developed with Ubuntu Linux, CUDA 9.x, and NVIDIA GeForce 940Mx.
#### Step1: Modify the ```Makefile```
```
GENCODE_FLAGS  = -gencode arch=compute_xx,code=sm_xx
```
Replace the ```xx``` with the correct gencode. Check with the [link](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)
#### Step2: Make
```
make out.ppm
```
### Result
todo
## Acknowledgement
The CPU-version ray tracer refers to the tutorial from [GAMES101](https://sites.cs.ucsb.edu/~lingqi/teaching/games101.html) a lots.

The GPU-version ray tracer is heavily based the [ray tracing in one weekend series](https://raytracing.github.io/books) and [NVIDIA CUDA tutorials](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)


