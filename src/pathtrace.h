#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void normalPass(int iterCount);
void albedoPass(int iterCount);
void roughnessPass(int iterCount); 
void metallicPass(int iterCount); 

