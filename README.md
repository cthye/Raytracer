# CSE167 Final Project Write Up

#### **This is a personal project implementing a ray tracer**

## Brief Introduction
This project implements a ray tracer from scratch. The reason why I gave up using the starter code in Homework3 lies on that OpenGL is basically a rasterizer and not so good for building ray tracer (although it can). Also, I want to implement some optimized features to speed up rendering (for example, building BVH). Therefore, I rewrite this renderer. 

This renderer aims to have the following features:
- Bounding Volume Hierachy
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
Define the scene of the world. It has the following methods and parameters:
- Frame(image) paramters: width / height
- Camera paramters: imageAspectRatio / fovy / eyePos / target / vup
- Cast ray to the viewport
- Add objects (meshTriangle, sphere, ect..) to scene and build BVH
- Calculate the intersections with objects in the scene
#### Casting ray to viewport
The viewport (aka screen) coordinate ranges from x[-1, 1] and y[-1, 1]. 
Given the imageAspectRatio and fovy, then 
$$viewportHeight = tan(\frac{fovy}{2}) * 2$$
$$viewportWidth = imageAspectRatio * viewportHeight$$
For pixel in (i, j), the center coordinate is 
$$x = 2 *\frac{i + 0.5}{2} - 1, y = 1 - \frac{i + 0.5}{2}$$

Given the eyePos, target, and vup vector, calculate 

![positonable camera](https://raytracing.github.io/images/fig-1.16-cam-view-up.jpg)

$$w = normalize(eyePos - target), u = normalize(cross(vup, w)), v = normalize(cross(w, u))$$

then the ray direction is 
$$normalize(x * viewportWidth * u + y * viewportHeight * v - w)$$

#### Mutiple samples Anti-aliasing
Sample the pixel multiple times and average the result by the end. For each ray in a single pixel, send it to different points near the center by adding a random float (range from 0 - 1) to the center coordinate. 

![MSAA](https://raytracing.github.io/images/fig-1.07-pixel-samples.jpg)

Now the ray's target coordinate is:
$$x = 2 *\frac{i + randomFloat }{2} - 1, y = 1 - \frac{i + randomFloat}{2}$$

### Build Bounding Volume Hierachy

If calculate the intersection for every single object, the time cost will be linear to the number of objects, which is the main time-bottleneck. Therefore, before doing the ray intersection part, build a BVH to speed up the procedure. 

#### Intersect with Axis-aligned Bounding Boxes (AABB)

Rather than intersect a ray with privimitive objects, intersect with bounding boxes is simpler. AABB is just the intersection of 3 axis-aliigned intervals. For a ray hit a box, first figure out whether the ray hits the plane intervals. 

![x coordinate](https://raytracing.github.io/images/fig-2.03-ray-slab.jpg)

If only consider x coordinates, let $$t_{min}$$ represents the time when the ray enters the left-side plane $$x = x_0$$ and  $$t_{max}$$ represents the time when the ray exits the right-side plane $$x = x_1$$. For a given ray $$R(t) = A + tb,$$

$$t_{min} = \frac{x_0 - A_x}{b_x}, t_{max} = \frac{x_1 - A_x}{b_x}$$

For 3 dimenstions, $$t_{enter} = max(t_{xmin}, t_{ymin}, t_{zmin}). t_{exit} = min(t_{xmax}, t_{ymax}, t_{zmax})$

Notice that both t_min and t_max can be negative, which means that the ray'sorigin is inside of the box or the ray does not enter the box. Therefore, if a ray hits the box, it needs to satisfy $$t_{enter} <= t_{exit}$$ and $$t_{exit} >= 0$$

#### Construct Bounding Boxes for Primitives

Two vectors (points) can specify a bounding box, a point pMin which represents the lower left corner and the other pMax represents the upper right corner. For each primitives, it has its own method to calculate the bouding box. For example, the bounding box of sphere is represented by 

$$pMin = center - (r, r, r), pMax = center + (r, r, r)$$. 

And for a triangle, 

$$pMin = min(x_1, x_2, x_3), min(y_1, y_2, y_3), min(z_1, z_2, z_3), pMin = max(x_1, x_2, x_3), max(y_1, y_2, y_3), max(z_1, z_2, z_3)$$. 

Creating a bounding box from 2 boxes by union is just update the pMin and pMax: 

 $$pMin = min(pMin1, pMin2), pMax = max(pMax1, pMax2)$$

#### Construct Bounding Boxes Hierachy for Object List

The main idea of constructing BVH is building a tree whose nodes contain the bounding boxes recursively.  Choose the max-extent axis from x/y/z, sort the primitives, and put half of them into left/right subtree. The leaf nodes only conatin one bounding box. 

#### Intersect with BHV



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

Also, most of the images in this writeup come from the ray tracing in one weekend series.
