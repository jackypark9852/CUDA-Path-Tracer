#include "intersections.h"
#include "utilities.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

// stable quadratic
__host__ __device__ inline bool solveQuadratic(float A, float B, float C, float& t0, float& t1)
{
    const float invA = 1.0f / A;
    B *= invA;
    C *= invA;

    const float neg_halfB = -0.5f * B;
    const float disc = neg_halfB * neg_halfB - C;
    if (disc < 0.0f) return false;

    const float u = sqrtf(disc);
    float r0 = neg_halfB - u;
    float r1 = neg_halfB + u;
    if (r0 > r1) { float tmp = r0; r0 = r1; r1 = tmp; }
    t0 = r0; t1 = r1;
    return true;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    const float radius = 0.5f;
    const float EPS = 1e-5f;

    const glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3       rd = multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f));
    rd = glm::normalize(rd);

    const float A = glm::dot(rd, rd);
    const float B = 2.0f * glm::dot(rd, ro);
    const float C = glm::dot(ro, ro) - radius * radius;

    float t0, t1;
    if (!solveQuadratic(A, B, C, t0, t1)) return -1.0f;

    float t_obj = (t0 > EPS) ? t0 : ((t1 > EPS) ? t1 : -1.0f);
    if (t_obj < 0.0f) return -1.0f;

    const glm::vec3 p_os = ro + t_obj * rd;
    const glm::vec3 n_os = glm::normalize(p_os); // sphere centered at origin

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(p_os, 1.0f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(n_os, 0.0f)));

    outside = (glm::dot(normal, r.direction) < 0.0f);
    if (!outside) normal = -normal;

    const float dirLen = glm::length(r.direction);
    if (dirLen <= 0.0f) return -1.0f;
    return glm::length(intersectionPoint - r.origin) / dirLen;
}


// From CIS561
// Moller Trumbore intersection
__host__ __device__ bool RayTriangleIntersect(
    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
    glm::vec3 rayOrigin, glm::vec3 rayDirection,
    float& outDist, glm::vec3& outBary) 
{
    glm::vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = p1 - p0;
    edge2 = p2 - p0;
    h = cross(rayDirection, edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) {
        return false;
    }
    f = 1.0 / a;
    s = rayOrigin - p0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = cross(s, edge1);
    v = f * dot(rayDirection, q);
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }

    float t = f * dot(edge2, q);
    if (t > EPSILON) {
        outDist = t; 
        glm::vec3 intersectionPoint = rayOrigin + rayDirection * t; 
        outBary = Barycentric(intersectionPoint, p0, p1, p2); 
        return true;
    }
    else
        return false;
}

__host__ __device__ glm::vec3 Barycentric(glm::vec3 p, glm::vec3 t1, glm::vec3 t2, glm::vec3 t3) {
    glm::vec3 edge1 = t2 - t1;
    glm:: vec3 edge2 = t3 - t2;
    float S = length(cross(edge1, edge2));

    edge1 = p - t2;
    edge2 = p - t3;
    float S1 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t3;
    float S2 = length(cross(edge1, edge2));

    edge1 = p - t1;
    edge2 = p - t2;
    float S3 = length(cross(edge1, edge2));

    return glm::vec3(S1 / S, S2 / S, S3 / S);
}

__host__ __device__ float RayAABBIntersection(
    const AABB& aabb,
    const Ray& r,
    float tMax)
{
    const glm::vec3 invDir = 1.0f / r.direction;
    const glm::vec3 t0 = (aabb.minBounds - r.origin) * invDir;
    const glm::vec3 t1 = (aabb.maxBounds - r.origin) * invDir;

    const float txmin = fminf(t0.x, t1.x), txmax = fmaxf(t0.x, t1.x);
    const float tymin = fminf(t0.y, t1.y), tymax = fmaxf(t0.y, t1.y);
    const float tzmin = fminf(t0.z, t1.z), tzmax = fmaxf(t0.z, t1.z);

    float tEnter = fmaxf(txmin, fmaxf(tymin, tzmin));
    float tExit = fminf(txmax, fminf(tymax, tzmax));

    if (tExit >= tEnter && tExit > 0.0f && tEnter < tMax) {
        return fmaxf(tEnter, 0.0f);
    }
    return -1.0;    
}
