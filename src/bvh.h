#pragma once

#include <glm/common.hpp>
#include <string>
#include <vector>

// BVH implementation inspired by:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct AABB
{
    glm::vec3 minBounds = glm::vec3(1e30);
    glm::vec3 maxBounds = glm::vec3(-1e30); 

    void reset() {
        minBounds = glm::vec3(1e30);
        maxBounds = glm::vec3(-1e30);
    }

    void grow(glm::vec3 pos) {
        for (uint32_t i = 0; i < 3; ++i) {
            minBounds[i] = fminf(minBounds[i], pos[i]);
            maxBounds[i] = fmaxf(maxBounds[i], pos[i]);
        }
    }

    void grow(const AABB& b) {
        for (uint32_t i = 0; i < 3; ++i) {
            minBounds[i] = fminf(minBounds[i], b.minBounds[i]);
            maxBounds[i] = fmaxf(maxBounds[i], b.maxBounds[i]);
        }
    }

    float area()
    {
        glm::vec3 e = maxBounds - minBounds; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x;
    }
};

struct Bin { AABB aabb; int triCount = 0; };

struct BvhNode
{
    AABB aabb; 
    unsigned int leftFirst, triCount;;
};

// Constructs BVH based on tri centrioids, outputs in outBvhNodes, 
// and rearranges indices array elements so that tris in the same bvh node are memory adjacent
bool ConstructBVH(
    const std::vector<glm::vec3>& positions, 
    std::vector<glm::vec3>& centroids,
    std::vector<uint32_t>& indices,             // rearranged  
    std::vector<BvhNode>& outBvhNodes,          // cleared and populated 
    std::string* err
);

void UpdateNodeBounds(
    uint32_t nodeIdx,
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    std::vector<BvhNode>& outBvhNodes
);

void FindSplitPlaneNaive(
    const BvhNode& currentNode,
    int& axis, 
    float& splitPos);

float EvaluateSAH(
    const BvhNode& currentNode,
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& centroids,
    const std::vector<uint32_t>& indices,
    int axis,
    float splitPos
);

float FindSplitPlaneSAH(
    const BvhNode& currentNode,
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& centroids,
    const std::vector<uint32_t>& indices,
    int& axis,
    float& splitPos
);

void Subdivide(
    uint32_t nodeIdx,
    const std::vector<glm::vec3>& positions,
    std::vector<glm::vec3>& centroids,
    std::vector<uint32_t>& indices,             // rearranged  
    std::vector<BvhNode>& outBvhNodes           // cleared and populated 
);