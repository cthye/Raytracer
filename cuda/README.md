# CSE167 Final Project Write Up

#### **This is a personal project implementing a ray tracer**

## Brief Introduction
This project implements a GPU-based ray tracer from scratch. The reason why I gave up using the starter code in Homework3 lies on that OpenGL is basically a rasterizer and not so convient for building ray tracer (although it can). But I don't want to lose the feature of parallel rendering under GPU. Therefore, so I write this renderer in CUDA. 

This renderer aims to have the following features:
- BVH SAH (?)
- MSAA antialiasing
- light 
- material: diffuse, metal, ...
- .......
- Parallel rendering with GPU
- stack (?)

## Explanation

This part is the explantion of relative maths and algorithms. 

...TODO

## Code Document

This part is the code document for describing how each module works.

#### Scene
Define the scene of the world.
- width / height / imageAspectRatio
- 

#### Renderer

#### Vector3f
Define ```Vector3f``` class for 3D vector operations

#### Utils
Define some useful global functions (clamp, deg2Rad, etc..)

...TODO

## Profile
...TODO

## Run the code
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
Render the result in PPM file. There are more tools to convert PPM to JPG (or other format).

## Acknowledgement
The renderer is heavily based the [ray tracing in one weekend series](https://raytracing.github.io/books) and [NVIDIA CUDA tutorials](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)


