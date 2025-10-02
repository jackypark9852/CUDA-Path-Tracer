#pragma once

#include <glm/common.hpp>
#include <string>
#include <vector>

// BVH implementation inspired by:
// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct BvhNode
{
    glm::vec3 aabbMin, aabbMax;
    unsigned int leftFirst, triCount;;
    const bool isLeaf() { return triCount > 0; }
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

void Subdivide(
    uint32_t nodeIdx,
    const std::vector<glm::vec3>& positions,
    std::vector<glm::vec3>& centroids,
    std::vector<uint32_t>& indices,             // rearranged  
    std::vector<BvhNode>& outBvhNodes           // cleared and populated 
);