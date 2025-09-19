#pragma once
#include <cstdint>

struct PathTraceSettings {
	bool enableStreamCompaction = true;
	bool enableMaterialSorting = true;
};

// just one instance
extern PathTraceSettings g_settings; 
