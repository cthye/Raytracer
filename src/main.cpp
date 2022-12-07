#include "Renderer.hpp"
#include "Scene.hpp"
#include "Triangle.hpp"
#include "Sphere.hpp"
#include "Vector.hpp"
#include "global.hpp"
#include "Camera.hpp"
#include <chrono>

int main(int argc, char** argv)
{
    // Scene scene(784, 784);
    int w = 512;
    int h = 512;
    Scene scene(w, h);
    
    float imageAspectRatio  = w/h;
    // float fovy = 15;
    float fovy = 19;

    Vector3f target = Vector3f(278.f, 273.f, -799.f); 
    Vector3f eye_pos = Vector3f(278., 273., -800.);
    Vector3f vup = Vector3f(0.f, 1.f, 0.f); 

    // Vector3f target = Vector3f(300.f, 472.5f, -399.f); 
    // Vector3f eye_pos = Vector3f(300., 473., -400.);
    // Vector3f vup = Vector3f(0.f, 1.f, 0.f); 
    Camera cam(imageAspectRatio, fovy, eye_pos, target, vup);

    std::shared_ptr<Material> red = std::make_shared<Material>(DIFFUSE, Vector3f(0.63f, 0.065f, 0.05f), Vector3f(0.0f));
    std::shared_ptr<Material> green = std::make_shared<Material>(DIFFUSE, Vector3f(0.14f, 0.45f, 0.091f), Vector3f(0.0f));
    std::shared_ptr<Material> white = std::make_shared<Material>(DIFFUSE, Vector3f(0.725f, 0.71f, 0.68f), Vector3f(0.0f));
    std::shared_ptr<Material> pink = std::make_shared<Material>(DIFFUSE, Vector3f(0.725f, 0.61f, 0.68f), Vector3f(0.0f));
    std::shared_ptr<Material> light = std::make_shared<Material>(DIFFUSE, Vector3f(0.65f), (8.0f * Vector3f(0.747f+0.058f, 0.747f+0.258f, 0.747f) + 15.6f * Vector3f(0.740f+0.287f,0.740f+0.160f,0.740f) + 18.4f *Vector3f(0.737f+0.642f,0.737f+0.159f,0.737f)));

    MeshTriangle floor("../models/cornellbox/floor.obj", "floor", white);
    MeshTriangle shortbox("../models/cornellbox/shortbox.obj", "shortbox", white);
    MeshTriangle tallbox("../models/cornellbox/tallbox.obj", "tallbox", white);
    MeshTriangle bunny("../models/bunny/bunny.obj", Vector3f(200, -60, 150), Vector3f(1500, 1500, 1500), Vector3f(-1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, -1), pink);
    MeshTriangle left("../models/cornellbox/left.obj", "left", red);
    MeshTriangle right("../models/cornellbox/right.obj", "right", green);
    MeshTriangle light_("../models/cornellbox/light.obj", "light", light);
    // Sphere ball(Vector3f(200, 100, 150), 100, pink);


    scene.Add(std::make_shared<MeshTriangle>(floor));
    scene.Add(std::make_shared<MeshTriangle>(shortbox));
    scene.Add(std::make_shared<MeshTriangle>(tallbox));
    scene.Add(std::make_shared<MeshTriangle>(left));
    scene.Add(std::make_shared<MeshTriangle>(right));
    scene.Add(std::make_shared<MeshTriangle>(light_));
    scene.Add(std::make_shared<MeshTriangle>(bunny));
    // scene.Add(std::make_shared<Sphere>(ball));


    scene.buildBVH();

    Renderer r;

    auto start = std::chrono::system_clock::now();
    r.Render(scene, cam);
    auto stop = std::chrono::system_clock::now();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}