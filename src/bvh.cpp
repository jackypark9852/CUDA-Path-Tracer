#include "bvh.h"

bool ConstructBVH(
	const std::vector<glm::vec3>& positions, 
	std::vector<glm::vec3>& centroids, 
	std::vector<uint32_t>& indices, 
	std::vector<BvhNode>& outBvhNodes, 
	std::string* err)
{
	if (indices.size() % 3 != 0) {
		if(err) *err += "bvh construction failed: indices not multiple of three";
		return false; 
	}

	size_t nTri = static_cast<size_t>(indices.size() / 3);
	outBvhNodes.clear(); 
	if (nTri == 0) return true;

	// Max number of nodes required is 2N - 1, 
	// but we pad 1 after root node for better memory coherany
	// since left and right child node (32 bytees each) can fit in 1 64 bytes cache 
	outBvhNodes.reserve(nTri * 2); 
	
	// assign all triangles to root node
	uint32_t rootNodeIdx = 0; 
	outBvhNodes.emplace_back();
	BvhNode& root = outBvhNodes[rootNodeIdx];
	outBvhNodes.emplace_back(); // padding for better caching
	root.leftFirst = 0, root.triCount = nTri;
	UpdateNodeBounds(rootNodeIdx, positions, indices, outBvhNodes);

	// subdivide recursively
	Subdivide(rootNodeIdx, positions, centroids, indices, outBvhNodes);
	return true;
}

void UpdateNodeBounds(
	uint32_t nodeIdx,
	const std::vector<glm::vec3>& positions,
	const std::vector<uint32_t>& indices,
	std::vector<BvhNode>& outBvhNodes)
{
	BvhNode& node = outBvhNodes[nodeIdx];
	AABB& aabb = node.aabb;
	aabb.minBounds = glm::vec3(FLT_MAX);
	aabb.maxBounds = glm::vec3(FLT_MIN);
	for (uint32_t i = node.leftFirst; i < node.leftFirst + node.triCount; i++)
	{
		glm::vec3 v0 = positions[indices[i * 3 + 0]];
		glm::vec3 v1 = positions[indices[i * 3 + 1]];
		glm::vec3 v2 = positions[indices[i * 3 + 2]];
		node.aabb.grow(v0);
		node.aabb.grow(v1);
		node.aabb.grow(v2);
	}
}

void FindSplitPlaneNaive(const BvhNode& currentNode, int& axis, float& splitPos) {
	// find split position and axis
	glm::vec3 extent = currentNode.aabb.maxBounds - currentNode.aabb.minBounds;
	axis = 0;
	if (extent.y > extent.x) axis = 1;
	if (extent.z > extent[axis]) axis = 2;
	if (extent[axis] <= 0.0f) return; // degenerate; make leaf

	splitPos = currentNode.aabb.minBounds[axis] + extent[axis] * 0.5f;
}

float EvaluateSAH(
	const BvhNode& currentNode,
	const std::vector<glm::vec3>& positions,
	const std::vector<glm::vec3>& centroids,
	const std::vector<uint32_t>& indices,
	int axis,
	float splitPos)
{
	// split prims 
	int start = static_cast<int>(currentNode.leftFirst);
	int end = start + static_cast<int>(currentNode.triCount);
	AABB leftBox{}, rightBox{};
	int leftCount = 0, rightCount = 0;
	for (uint32_t i = start; i < end; ++i) {
		glm::vec3 v0 = positions[indices[i * 3 + 0]];
		glm::vec3 v1 = positions[indices[i * 3 + 1]];
		glm::vec3 v2 = positions[indices[i * 3 + 2]];

		AABB& box = (centroids[i][axis] < splitPos) ? leftBox : rightBox;
		int& count = (centroids[i][axis] < splitPos) ? leftCount : rightCount;

		count++;
		box.grow(v0);
		box.grow(v1);
		box.grow(v2);
	}

	float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
	return cost > 0 ? cost : 1e30;
}

#define BINS 8

float FindSplitPlaneSAH(
	const BvhNode& currentNode, 
	const std::vector<glm::vec3>& positions, 
	const std::vector<glm::vec3>& centroids, 
	const std::vector<uint32_t>& indices, 
	int& axis, 
	float& splitPos)
{
	float bestCost = 1e30f;

	// iteratively find the best split plane
	for (int candidateAxis = 0; candidateAxis < 3; candidateAxis++) {
		float boundsMin = 1e30f, boundsMax = -1e30f;
		int start = static_cast<int>(currentNode.leftFirst);
		int end = start + static_cast<int>(currentNode.triCount);
		for (uint32_t i = start; i < end; ++i) {
			glm::vec3 centroid = centroids[i];
			boundsMin = fminf(boundsMin, centroid[candidateAxis]);
			boundsMax = fmaxf(boundsMax, centroid[candidateAxis]);
 		}
		if (boundsMin == boundsMax) continue;

		// populate bins
		Bin bin[BINS];
		float scale = BINS / (boundsMax - boundsMin);
		for (uint32_t i = start; i < end; ++i)
		{
			int binIdx = glm::min(BINS - 1,
				(int)((centroids[i][candidateAxis] - boundsMin) * scale));
			bin[binIdx].triCount++;

			glm::vec3 v0 = positions[indices[i * 3 + 0]];
			glm::vec3 v1 = positions[indices[i * 3 + 1]];
			glm::vec3 v2 = positions[indices[i * 3 + 2]];
		
			bin[binIdx].aabb.grow(v0);
			bin[binIdx].aabb.grow(v1);
			bin[binIdx].aabb.grow(v2);
		}

		float leftArea[BINS - 1], rightArea[BINS - 1];
		int leftCount[BINS - 1], rightCount[BINS - 1];
		AABB leftBox, rightBox;
		int leftSum = 0, rightSum = 0;
		for (int i = 0; i < BINS - 1; i++)
		{
			leftSum += bin[i].triCount;
			leftCount[i] = leftSum;
			leftBox.grow(bin[i].aabb);
			leftArea[i] = leftBox.area();
			rightSum += bin[BINS - 1 - i].triCount;
			rightCount[BINS - 2 - i] = rightSum;
			rightBox.grow(bin[BINS - 1 - i].aabb);
			rightArea[BINS - 2 - i] = rightBox.area();
		}
		
		scale = (boundsMax - boundsMin) / BINS;
		for (int i = 0; i < BINS - 1; i++)
		{
			float planeCost =
				leftCount[i] * leftArea[i] + rightCount[i] * rightArea[i];
			if (planeCost < bestCost)
				axis = candidateAxis, splitPos = boundsMin + scale * (i + 1),
				bestCost = planeCost;
		}
	}

	return bestCost; 
}

#ifndef MIN_TRI_COUNT
#define MIN_TRI_COUNT 2
#endif

void Subdivide(
	uint32_t nodeIdx,
	const std::vector<glm::vec3>& positions,
	std::vector<glm::vec3>& centroids,
	std::vector<uint32_t>& indices,
	std::vector<BvhNode>& outBvhNodes           // cleared and populated 
) {
	BvhNode& currentNode = outBvhNodes[nodeIdx];
	if (currentNode.triCount < 2u * static_cast<uint32_t>(MIN_TRI_COUNT)) {
		return; // keep as leaf to respect the minimum per leaf
	}


	// find split position and axis
	int axis = 0; 
	float splitPos = 0.0f; 

	//FindSplitPlaneNaive(currentNode, axis, splitPos);
	float bestCost = FindSplitPlaneSAH(currentNode, positions, centroids, indices, axis, splitPos); 
	float parentCost = static_cast<float>(currentNode.triCount) * outBvhNodes[nodeIdx].aabb.area();
	if (bestCost >= parentCost) return;

	// split prims 
	int i = static_cast<int>(currentNode.leftFirst);
	int j = i + static_cast<int>(currentNode.triCount) - 1;
	while (i <= j)
	{
		if (centroids[i][axis] < splitPos)
			i++;
		else {
			std::swap(centroids[i], centroids[j]);
			std::swap(indices[i * 3 + 0], indices[j * 3 + 0]);
			std::swap(indices[i * 3 + 1], indices[j * 3 + 1]);
			std::swap(indices[i * 3 + 2], indices[j * 3 + 2]);
			j--;
		}
	}

	// current node is a leaf node 
	int leftCount = i - currentNode.leftFirst;
	if (leftCount == 0 || leftCount == currentNode.triCount) return;

	// create child nodes
	uint32_t leftChildIdx = static_cast<uint32_t>(outBvhNodes.size()); 
	outBvhNodes.emplace_back(); 
	uint32_t rightChildIdx = static_cast<uint32_t>(outBvhNodes.size());
	outBvhNodes.emplace_back(); 

	BvhNode& leftChild = outBvhNodes[leftChildIdx];
	leftChild.leftFirst = currentNode.leftFirst;
	leftChild.triCount = leftCount;
	UpdateNodeBounds(leftChildIdx, positions, indices, outBvhNodes);

	BvhNode& rightChild = outBvhNodes[rightChildIdx]; 
	rightChild.leftFirst = i;
	rightChild.triCount = currentNode.triCount - leftCount;
	UpdateNodeBounds(rightChildIdx, positions, indices, outBvhNodes);

	// mark current node as non-leaf node
	currentNode.leftFirst = leftChildIdx;
	currentNode.triCount = 0;

	// recursively subdivide
	Subdivide(leftChildIdx, positions, centroids, indices, outBvhNodes); 
	Subdivide(rightChildIdx, positions, centroids, indices, outBvhNodes);
}