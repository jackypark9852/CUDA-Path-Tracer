#pragma once
// cuda timing helper (accumulates totals across calls/iterations)

#include <unordered_map>
#include <string>

struct CudaTimingStat {
    double sum_ms = 0.0;
    int    count = 0;
};

// control / query
void cuda_timing_reset();
void cuda_timing_accum(const char* tag, float ms);

// dumps totals (sorted by total_ms desc). Clears the table if reset_after=true.
void cuda_timing_dump_totals(bool reset_after = false);

inline void cuda_timing_dump_totals_and_reset() { cuda_timing_dump_totals(true); }

// time a single expression (kernel launch, thrust call, etc.)
// usage:
//   TIME_CUDA("intersect",
//       (ComputeIntersections<<<grid, block>>>(...)));
#define TIME_CUDA(TAG, CALL_EXPR)                                                      \
    do {                                                                               \
        cudaEvent_t __ct_start, __ct_stop;                                             \
        cudaEventCreate(&__ct_start);                                                  \
        cudaEventCreate(&__ct_stop);                                                   \
        cudaEventRecord(__ct_start);                                                   \
        CALL_EXPR;                                                                     \
        cudaEventRecord(__ct_stop);                                                    \
        cudaEventSynchronize(__ct_stop);                                               \
        float __ct_ms = 0.0f;                                                          \
        cudaEventElapsedTime(&__ct_ms, __ct_start, __ct_stop);                         \
        cuda_timing_accum(TAG, __ct_ms);                                               \
        cudaEventDestroy(__ct_start);                                                  \
        cudaEventDestroy(__ct_stop);                                                   \
    } while (0)
