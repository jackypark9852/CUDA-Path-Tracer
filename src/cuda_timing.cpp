#include "cuda_timing.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>

static std::unordered_map<std::string, CudaTimingStat> g_ct_table;

void cuda_timing_reset() {
    g_ct_table.clear();
}

void cuda_timing_accum(const char* tag, float ms) {
    const std::string key = tag ? std::string(tag) : std::string("");
    auto& s = g_ct_table[key];
    s.sum_ms += static_cast<double>(ms);
    s.count += 1;
}

void cuda_timing_dump_totals(bool reset_after) {
    if (g_ct_table.empty()) {
        std::printf("\n==== cuda timings (totals) ====\n(no entries)\n\n");
        return;
    }

    size_t name_w = 12;
    for (const auto& kv : g_ct_table) name_w = std::max(name_w, kv.first.size());

    std::vector<std::pair<std::string, CudaTimingStat>> rows;
    rows.reserve(g_ct_table.size());
    for (const auto& kv : g_ct_table) rows.emplace_back(kv.first, kv.second);

    std::sort(rows.begin(), rows.end(),
        [](const auto& a, const auto& b) { return a.second.sum_ms > b.second.sum_ms; });

    double grand_ms = 0.0;
    for (const auto& r : rows) grand_ms += r.second.sum_ms;

    // header
    const int w_total = (int)(name_w + 3 + 12 + 3 + 8 + 3 + 8);
    std::printf("\n==== cuda timings (totals) ====\n");
    std::printf("%-*s | %12s | %8s | %8s\n",
        (int)name_w, "name", "total_ms", "calls", "% total");
    for (int i = 0; i < w_total; ++i) std::printf("-");
    std::printf("\n");

    // rows
    for (const auto& [name, st] : rows) {
        double pct = (grand_ms > 0.0) ? (100.0 * st.sum_ms / grand_ms) : 0.0;
        std::printf("%-*s | %12.3f | %8d | %7.2f%%\n",
            (int)name_w, name.c_str(), st.sum_ms, st.count, pct);
    }
    for (int i = 0; i < w_total; ++i) std::printf("-");
    std::printf("\n%-*s | %12.3f | %8s | %7s\n\n",
        (int)name_w, "grand total", grand_ms, "", "");

    if (reset_after) g_ct_table.clear();
}
