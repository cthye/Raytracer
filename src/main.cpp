#include "Renderer.hpp"
#include "Scene.hpp"
#include "Triangle.hpp"
// #include "Sphere.hpp"
#include "Vector.hpp"
#include "global.hpp"
#include <chrono>

// In the main function of the program, we create the scene (create objects and
// lights) as well as set the options for the render (image width and height,
// maximum recursion depth, field-of-view, etc.). We then call the render
// function().
int main(int argc, char** argv)
{

    // Change the definition here to change resolution
    // Scene scene(784, 784);
    Scene scene(512, 512);

    Material* red = new Material(DIFFUSE, Vector3f(0.0f));
    // Material* red = new Material(MICROFACET, Vector3f(0.0f));
    red->Kd = Vector3f(0.63f, 0.065f, 0.05f);
    Material* green = new Material(DIFFUSE, Vector3f(0.0f));
    // Material* green = new Material(MICROFACET, Vector3f(0.0f));
    green->Kd = Vector3f(0.14f, 0.45f, 0.091f);
    Material* white = new Material(DIFFUSE, Vector3f(0.0f));
    white->Kd = Vector3f(0.725f, 0.71f, 0.68f);
    Material* pink = new Material(DIFFUSE, Vector3f(0.0f));
    pink->Kd = Vector3f(0.725f, 0.61f, 0.68f);
    Material* light = new Material(DIFFUSE, (8.0f * Vector3f(0.747f+0.058f, 0.747f+0.258f, 0.747f) + 15.6f * Vector3f(0.740f+0.287f,0.740f+0.160f,0.740f) + 18.4f *Vector3f(0.737f+0.642f,0.737f+0.159f,0.737f)));
    light->Kd = Vector3f(0.65f);

    MeshTriangle floor("../models/cornellbox/floor.obj", "floor", white);
    MeshTriangle shortbox("../models/cornellbox/shortbox.obj", "shortbox", white);
    MeshTriangle tallbox("../models/cornellbox/tallbox.obj", "tallbox", white);
    MeshTriangle bunny("../models/bunny/bunny.obj", Vector3f(200, -60, 150), Vector3f(1500, 1500, 1500), Vector3f(-1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, -1), pink);
    MeshTriangle left("../models/cornellbox/left.obj", "left", red);
    MeshTriangle right("../models/cornellbox/right.obj", "right", green);
    MeshTriangle light_("../models/cornellbox/light.obj", "light", light);

    // MeshTriangle floor("../models/cornellbox/floor.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), white);
    // MeshTriangle shortbox("../models/cornellbox/shortbox.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), white);
    // MeshTriangle tallbox("../models/cornellbox/tallbox.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), white);
    // MeshTriangle bunny("../models/bunny/bunny.obj", Vector3f(0, -100, -300), Vector3f(1500, 1500, 1500), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), pink);
    // MeshTriangle left("../models/cornellbox/left.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), red);
    // MeshTriangle right("../models/cornellbox/right.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), green);
    // MeshTriangle light_("../models/cornellbox/light.obj", Vector3f(-250, -250, -600), Vector3f(1, 1, 1), Vector3f(1, 0, 0), Vector3f(0, 1, 0), Vector3f(0, 0, 1), light);


    scene.Add(std::make_shared<MeshTriangle>(floor));
    scene.Add(std::make_shared<MeshTriangle>(shortbox));
    scene.Add(std::make_shared<MeshTriangle>(tallbox));
    scene.Add(std::make_shared<MeshTriangle>(left));
    scene.Add(std::make_shared<MeshTriangle>(right));
    scene.Add(std::make_shared<MeshTriangle>(light_));
    scene.Add(std::make_shared<MeshTriangle>(bunny));

    scene.buildBVH();

    Renderer r;

    auto start = std::chrono::system_clock::now();
    r.Render(scene);
    auto stop = std::chrono::system_clock::now();

    std::cout << "Render complete: \n";
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::hours>(stop - start).count() << " hours\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::minutes>(stop - start).count() << " minutes\n";
    std::cout << "          : " << std::chrono::duration_cast<std::chrono::seconds>(stop - start).count() << " seconds\n";

    return 0;
}