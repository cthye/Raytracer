#include "Scene.hpp"

void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects); // improve by using SAH
}

Intersection Scene::intersect(const Ray &ray) const {
    return this->bvh->Intersect(ray);
}

//* pos得到的是被采样的光源坐标，pdf可以理解为1/A
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

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray &ray, int depth) const {
    // TODO Implement Path Tracing Algorithm here
    Intersection intersection = Scene::intersect(ray);
    Object *hitObject = intersection.obj;
    Vector3f hitColor = this->backgroundColor;

    if (intersection.happened) {
        if (intersection.obj->getName() == "light") {
            return Vector3f(1.);
        }
        hitColor = shader(intersection, -ray.direction);
    }

    return hitColor;
}

Vector3f Scene::shader(Intersection intersection, Vector3f wo) const {
    Vector3f L_dir(0., 0., 0.);
    Vector3f L_indir(0., 0., 0.);

    Vector3f hitPoint = intersection.coords;
    Vector3f N = intersection.normal;
    Material *m = intersection.m;

    //* uniformaly sample the light
    Intersection xx;
    float pdf;
    Scene::sampleLight(xx, pdf);

    Vector3f lightDir = normalize(xx.coords - hitPoint);  // p -> light
    float dist2Light = Vector3f::secondNorm(xx.coords, hitPoint);

    Vector3f shadowOrig;
    
    //* avoid the object shadow itself and pull the shadowPoint forward the light a bit
    shadowOrig = hitPoint - 0.0001 * lightDir;
    // shadowOrig = hitPoint;

    Ray shadowRay(shadowOrig, lightDir); 
    Intersection intersWithLight = bvh->Intersect(shadowRay);
    float dist = Vector3f::secondNorm(intersWithLight.coords, hitPoint);
    
    // if (!strcmp(intersection.obj->getName(), "tallbox")) {
    //     std::cout << "===========================hit tall box======================" << std::endl;
    // }


    // bool isNotBlocked = dist >= dist2Light - EPSILON;
    bool isNotBlocked = sqrt((intersWithLight.coords - xx.coords).norm()) < 0.2; //* 我真的醉了，得设置这么大的epsilon才能去掉横向黑线！


    // if (!strcmp(intersection.obj->getName(), "tallbox")) {
    //     // std::cout << "hit tall box" << std::endl;
        
    //     std::cout << "light: " << xx.coords << " hit:" << hitPoint
    //               << " happended:" << tmp.happened << " inser obj: " << tmp.obj
    //               << " light obj : " << xx.obj << " light emit:" << xx.emit
    //               << std::endl;

    //     std::cout << "block: " << isNotBlocked << " coords:" << tmp.coords
    //               << std::endl;
    //     // if (tmp.happened)
    //     //     std::cout << " emit: " << tmp.obj->hasEmit() << std::endl;
    //     if(!isNotBlocked) {
    //         std::cout << "blocked it self!" << std::endl;
    //     }
    // }

    if (isNotBlocked) {
        Vector3f wi = -lightDir;  // light -> p
        //* assume all directions are pointing outwards
        L_dir = xx.emit * m->eval(-wi, wo, N) * dotProduct(lightDir, N) *
                dotProduct(wi, xx.normal) /
                Vector3f::secondNorm(hitPoint, xx.coords) /
                std::max(pdf, EPSILON);
    }

    if (get_random_float() < RussianRoulette) {
        //* uniformaly choose a w_i (p->q)
        Vector3f wo_ = m->sample(wo, N);  // p -> q
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