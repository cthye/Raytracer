#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <stdio.h>

#include "ray.hpp"
#include "object.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "intersection.hpp"
#include "vector.hpp"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void rand_init(curandState *rand_state) {
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    curand_init(1984, 0, 0, rand_state);
}

__global__ void render_init(Scene **d_scene, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= (*d_scene)->width) || (j >= (*d_scene)->height)) return;
    int pixel_index = j * (*d_scene)->width + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

//todo: russian rollete & recursive
__device__ Vector3f shade(const Ray& r, Scene **scene, curandState *local_rand_state) {
    Vector3f cur_attenuation = Vector3f(1.0,1.0,1.0);
    Ray cur_ray = r;

    for(int i = 0; i < 50; i++) {
        Intersection rec;
        if((*scene)->hit(cur_ray, 0.001, FLT_MAX, rec)) {
            Ray scattered;
            Vector3f attenuation;
            // vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
            if(rec.m->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                // return emitted + attenuation * shade(scattered, world, depth + 1, state);
                // return emitted + attenuation * shade(scattered, world, depth + 1, state);
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                // printf("fail scatter: %f %f %f", r.direction.x, r.direction.y, r.direction.z);
                return Vector3f(0.0f, 0.0f, 0.0f);
            }
        } else {
            //background color
            Vector3f unit_direction = normalize(cur_ray.direction);
            float t = 0.5f*(unit_direction.y + 1.0f);
            Vector3f c = (1.0f-t)*Vector3f(1.0, 1.0, 1.0) + t*Vector3f(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    // printf("exceed: %f %f %f\n", r.direction.x, r.direction.y, r.direction.z);
    return Vector3f(0.0f, 0.0f, 0.0f);

    // Vector3f unit_direction = normalize(cur_ray.direction);
    // float t = 0.5f*(unit_direction.y + 1.0f);
    // printf("x': %f y': %f z': %f x: %f y: %f z: %f t: %f\n", cur_ray.direction.x, cur_ray.direction.y, cur_ray.direction.z, unit_direction.x, unit_direction.y, unit_direction.z, t);

    // Vector3f c = (1.0f-t)*Vector3f(1.0, 1.0, 1.0) + t*Vector3f(0.5, 0.7, 1.0);
    // printf("exceed0: %f %f %f\n", c.x, c.y, c.z);
    // c = cur_attenuation * c;
    // printf("exceed1: %f %f %f\n", c.x, c.y, c.z);
    // return c;
} 

__global__ void render(Vector3f* frameBuffer, Scene** scene, int spp, Camera** cam, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    // if(i == 0) printf("render %d %d", i, j);

    int max_x = (*scene)->width;
    int max_y = (*scene)->height;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    Vector3f col(0,0,0);
    for(int s=0; s < spp; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->getRay(u, v);
        col += shade(r, scene, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(spp);
    //gammar
    frameBuffer[pixel_index] = Vector3f(sqrt(col[0]),sqrt(col[1]), sqrt(col[2]));
}

#define RND (curand_uniform(local_rand_state))

__global__ void build_scene(Scene **d_scene, Object **d_objlist, Camera **d_camera, curandState* rand_state, int nx, int ny, int cnt) {
    if(threadIdx.x != 0 || blockIdx.x != 0) return;
    *d_camera = new Camera(Vector3f(13, 2, 3), Vector3f(0, 0, 0), Vector3f(0, 1, 0), 30.0f, float(nx)/float(ny));
    *d_scene  = new Scene(nx, float(nx)/float(ny), d_objlist, cnt);
    (*d_scene)->Add(new Sphere(Vector3f(0, -1000.0, -1), 1000, new Diffuse(Vector3f(0.5, 0.5, 0.5))));

    curandState* local_rand_state = rand_state;
    
    for(int a = -11; a < 11; a++) {
        for(int b = -11; b < 11; b++) {
            float choose_mat = RND;
            Vector3f center(a+RND,0.2,b+RND);
            if(choose_mat < 0.8f) {
                (*d_scene)->Add(new Sphere(center, 0.2,
                                            new Diffuse(Vector3f(RND*RND, RND*RND, RND*RND))));
            }
            else if(choose_mat < 0.95f) {
                (*d_scene)->Add(new Sphere(center, 0.2,
                                            new Metal(Vector3f(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND)));
            }
            else {
                (*d_scene)->Add(new Sphere(center, 0.2, new Dielectric(1.5)));
            }
        }
    }
    (*d_scene)->Add(new Sphere(Vector3f(0, 1,0),  1.0, new Dielectric(1.5)));
    (*d_scene)->Add(new Sphere(Vector3f(-4, 1, 0), 1.0, new Diffuse(Vector3f(0.4, 0.2, 0.1))));
    (*d_scene)->Add(new Sphere(Vector3f(4, 1, 0),  1.0, new Metal(Vector3f(0.7, 0.6, 0.5), 0.0)));
    rand_state = local_rand_state;
}

__global__ void free_world(Object **d_objList, Scene **d_world, Camera **d_camera) {
    for(int i=0; i < (*d_world)->objCnt; i++) {
        delete ((Object *)d_objList[i])->m;
        delete d_objList[i];
    }
    delete *d_world;
    delete *d_camera;
}


int main() {
     
    /* ==== define the scene ===== */
   
    // *d_scene = new Scene(1200, 3.0f/2.0f);
    
    int nx = 1200;
    int ny = 900;
    //  int nx = 400;
    // int ny = 300;
    // sample per pixels
    int spp =20; 
    // tx * ty blocks
    int tx = 8;  
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << spp << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //* allocate frame buffer
    int numPixels = nx * ny;
    size_t fbSize = numPixels * sizeof(Vector3f);
    Vector3f* frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, fbSize));

    //* allocate random state for every thread
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, numPixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //* allocate camera
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));

    //* allocate objects
    //todo: need to know how many object will be created in ahead...how to solve
    Scene **d_scene;
    checkCudaErrors(cudaMalloc((void **)&d_scene, sizeof(Scene *)));
    Object **d_objlist;
    int objCnt = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_objlist, objCnt*sizeof(Object *)));
    build_scene<<<1, 1>>>(d_scene, d_objlist, d_camera, d_rand_state2, nx, ny, objCnt);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    /* ==== render the scene ===== */
    
    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(d_scene, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(frameBuffer, d_scene, spp, d_camera, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::ofstream out("out.ppm");
    out << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            // std::cerr << j << ' ' << i << ' ' << pixel_index << ' ' << frameBuffer[pixel_index] << std::endl;

            // int ir = int(255.99*clamp(0.0f, 1.0f, frameBuffer[pixel_index].x));
            // int ig = int(255.99*clamp(0.0f, 1.0f, frameBuffer[pixel_index].y));
            // int ib = int(255.99*clamp(0.0f, 1.0f, frameBuffer[pixel_index].z));
            int ir = int(255.99*frameBuffer[pixel_index].x);
            int ig = int(255.99*frameBuffer[pixel_index].y);
            int ib = int(255.99*frameBuffer[pixel_index].z);
            out << ir << " " << ig << " " << ib << "\n";
        }
    }

    std::cerr << "done.\n";

    /* ==== free up memory ==== */

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_objlist,d_scene,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_scene));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_objlist));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(frameBuffer));

    cudaDeviceReset();
}