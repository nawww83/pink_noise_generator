#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <cmath>
#include "plotter.h"

namespace plotter_utils {

using u64 = uint64_t;
using i64 = int64_t;
using u32 = uint32_t;
using u16 = uint16_t;

template<int x>
inline constexpr auto is_prime() {
    int d = 2;
    bool ok = true;
    while (d < x) {
        ok &= ((x % d) != 0);
        d += 1;
    }
    return ok;
}

inline constexpr u64 sqrt_int(u64 x) {
    return static_cast<i64>(std::sqrt(x));
}

inline auto factor(u64 x) {
    std::map<u64, int> res{};
    auto inner_loop = [&res](u64 d, u64& x) -> bool {
        bool x_changed = false;
        while ((x % d == 0) && (x > 1)) {
            x /= d;
            res[d]++;
            x_changed = true;
        }
        return x_changed;
    };
    constexpr auto max_d = [](u64 x) -> u64 {
        return sqrt_int(x);
    };
    inner_loop(2, x);
    u64 d = 3;
    u64 md = max_d(x);
    while (d <= md) {
        md = inner_loop(d, x) ? max_d(x) : md;
        d += 2;
        d += (d % 3 == 0) ? 2 : 0;
    }
    if (x > 1 || res.empty()) {
        res[x]++;
    }

    return res;
}

inline auto divisors(u64 x) {
    const auto multipliers = factor(x);
    std::set<u64> divisors{1}; // must be sorted!
    for (const auto [t, c] : multipliers) {
        u64 T_ = t;
        std::set<u64> div_tmp{};
        for (int i=0; i<c; ++i) {
            div_tmp.insert(T_);
            T_ *= t;
        }
        std::set<u64> d_copy = divisors;
        for (auto d1 : d_copy) {
            for (auto d2 : div_tmp) {
                divisors.insert(d1*d2);
            }
        }
    }
    return divisors;
}

inline void AdjustY(PlotSettings& settings, double minY, double maxY) {
    settings.minY = minY;
    settings.maxY = maxY;
    const int minYTicks = 2;
    const int maxYTicks = 12;
    int scales = 0;
    while (std::max(std::abs(maxY), std::abs(minY)) < 100) {
        maxY = std::round(maxY);
        minY = std::round(minY);
        maxY *= 10;
        minY *= 10;
        scales++;
    }
    while (std::max(std::abs(maxY), std::abs(minY)) > 2000) {
        maxY /= 10;
        minY /= 10;
        maxY = std::round(maxY);
        minY = std::round(minY);
        scales--;
    }
    maxY = std::round(maxY);
    minY = std::round(minY);
    int numYTicks = -1;
    int swap_p = 0;
    for (;;) {
        const auto R = maxY - minY;
        auto divisors = plotter_utils::divisors(R);
        for (auto divisor : divisors) {
            if (divisor < 2) continue;
            if (scales > 0) {
                if ((divisor % 5) == 0) {
                    continue;
                }
                if ((divisor % 2) == 0) {
                    continue;
                }
            }
            // qDebug() << divisor << ",";
            if (divisor >= minYTicks && divisor <= maxYTicks) {
                numYTicks = divisor;
            }
        }
        if (numYTicks > 0) break;
        swap_p++;
        swap_p %= 2;
        if (swap_p) {
            maxY += 5;
        } else {
            minY -= 5;
        }
    }
    while (scales > 0) {
        maxY /= 10;
        minY /= 10;
        scales--;
    }
    while (scales < 0) {
        maxY *= 10;
        minY *= 10;
        scales++;
    }
    settings.minY = minY;
    settings.maxY = maxY;
    settings.numYTicks = std::max(numYTicks, minYTicks);
}

}
