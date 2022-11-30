cmake_minimum_required(VERSION 3.10)
project(RayTracing)

set(CMAKE_CXX_STANDARD 17)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  # set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/15.0.0/bin/clang++")
  set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/15.0.2/bin/clang++")
  set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -fopenmp")
endif()


find_package(Threads REQUIRED)

add_executable(RayTracing main.cpp Object.hpp Vector.cpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp
        Scene.hpp Light.hpp AreaLight.hpp BVH.cpp BVH.hpp Bounds3.hpp Ray.hpp Material.hpp Intersection.hpp
        Renderer.cpp Renderer.hpp)

target_link_libraries(RayTracing PRIVATE Threads::Threads)