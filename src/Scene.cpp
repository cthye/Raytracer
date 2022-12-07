#include "Scene.hpp"

void Scene::buildBVH() {
    // this->bvh = new BVHAccel(objects); 
    this->bvh = new BVHAccel(objects, BVHAccel::SplitMethod::SAH); // improve by using SAH
}

Intersection Scene::intersect(const Ray &ray) const {
    return this->bvh->Intersect(ray);
}

//* pos得到的是被采样的光源坐标，pdf=1/A
//* 注意光源也是一个mesh triangles object
void Scene::sampleLight(Intersection &pos, float &pdf) const {
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()) {
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()) {
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum) {
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

Vector3f Scene::castRay(const Ray &ray) const {
    Intersection intersection = Scene::intersect(ray);
    Vector3f hitColor = this->backgroundColor;

    if (intersection.happened) {
        hitColor = shader(intersection, -ray.direction);
    }

    return hitColor;
}

Vector3f Scene::shader(Intersection intersection, Vector3f wo) const {
    Vector3f L_dir(0., 0., 0.);
    Vector3f L_indir(0., 0., 0.);

    Vector3f hitPoint = intersection.coords;
    Vector3f N = intersection.normal;
    std::shared_ptr<Material> m = intersection.m;

    //* uniformaly sample the light
    Intersection lightPos;
    float pdf;
    Scene::sampleLight(lightPos, pdf);

    Vector3f lightDir = normalize(lightPos.coords - hitPoint);  // p -> light
    float dist2Light = Vector3f::secondNorm(lightPos.coords, hitPoint);

    //* avoid the object shadow itself and pull the shadowPoint forward the light a bit
    Vector3f shadowOrig;
    shadowOrig = hitPoint - 0.0001 * lightDir;

    Ray shadowRay(shadowOrig, lightDir); 
    Intersection intersWithLight = bvh->Intersect(shadowRay);
    float dist = Vector3f::secondNorm(intersWithLight.coords, hitPoint);
    
    // use a small epsilon to avoid floating point precision error
    bool isNotBlocked = sqrt((intersWithLight.coords - lightPos.coords).norm()) < 0.2;

    if (isNotBlocked) {
        Vector3f wi = -lightDir;  // light -> p
        //* assume all directions are pointing outwards
        L_dir = lightPos.emit * m->eval(-wi, wo, N) * dotProduct(lightDir, N) *
                dotProduct(wi, lightPos.normal) /
                Vector3f::secondNorm(hitPoint, lightPos.coords) /
                std::max(pdf, EPSILON);
    }

    if (get_random_float() < RussianRoulette) {
        Vector3f wo_ = m->scatter(N);  // p -> q
        float pdf = m->pdf(wo, wo_, N);

        Ray r(hitPoint + EPSILON * wo_, normalize(wo_));
        Intersection q = bvh->Intersect(r);
        if (q.happened && !q.m->hasEmission()) {
            L_indir = shader(q, -wo_) * m->eval(wo_, wo, N) *
                      dotProduct(wo_, N) / std::max(pdf, EPSILON) /
                      RussianRoulette;

        }
        
    }

    return L_dir + L_indir;
}