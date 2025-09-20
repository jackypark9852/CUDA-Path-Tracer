#pragma once
#include "../sceneStructs.h"
#include "../intersections.h"
#include "../interactions.h"

__global__ void shadePbrMaterial(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials);

__global__ void shadePerfectMaterial(
    int iter, int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials);
