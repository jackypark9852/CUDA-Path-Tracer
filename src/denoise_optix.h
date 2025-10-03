#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <vector>
#include <glm/glm.hpp>

void OptixDenoiseVectors(
    int width, int height,
    const std::vector<glm::vec3>& color,                                // required
    std::vector<glm::vec3>& denoised,
    const std::vector<glm::vec3>* albedo = nullptr,                     // optional
    const std::vector<glm::vec3>* normal = nullptr,                     // optional
    OptixDenoiserModelKind model = OPTIX_DENOISER_MODEL_KIND_LDR,       // LDR by default
    float blend = 0.0f                                                  // 0=full denoise, 1=original
);