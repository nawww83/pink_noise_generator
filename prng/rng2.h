#pragma once

/*
Simplified Random numbers generator with ~64 bit period.
*/

#include "lfsr.h"

#include <utility>
#include <cmath>

namespace rng2 {

using u16 = lfsr8::u16;
using u32 = lfsr8::u32;
using u64 = lfsr8::u64;
using u128 = std::pair<lfsr8::u64, lfsr8::u64>;

static constexpr u32 p1 = 23;
static constexpr u32 p2 = 19;
static constexpr int m = 4;

using LFSR_1 = lfsr8::LFSR<p1, m>;
using LFSR_2 = lfsr8::LFSR<p1, m>;
using LFSR_3 = lfsr8::LFSR<p2, m>;
using LFSR_4 = lfsr8::LFSR<p2, m>;
using STATE = lfsr8::u32x4;

inline STATE operator^(const STATE& x, const STATE& y) {
    STATE st;
    for (int i=0; i<m; ++i) {
        st[i] = x[i] ^ y[i];
    }
    return st;
}

inline void operator^=(STATE& x, const STATE& y) {
    for (int i=0; i<m; ++i) {
        x[i] ^= y[i];
    }
}

inline STATE operator%(const STATE& x, u32 p) {
    STATE st;
    for (int i=0; i<m; ++i) {
        st[i] = x[i] % p;
    }
    return st;
}

inline void operator%=(STATE& x, u32 p) {
    for (int i=0; i<m; ++i) {
        x[i] %= p;
    }
}

// K1: (1, 2, 5, 0), K2: (2, 2, 4, 1); T free: 12166, 528; (i, j)-pattern: (0, 3).  // p1=23
// K1: (1, 3, 10, 2), K2: (2, 2, 0, 4); T free: 2286, 3429; (i, j)-pattern: (2, 1). // p2=19
static constexpr STATE K1 = {1, 2, 5, 0};    // p1=23
static constexpr STATE K2 = {2, 2, 4, 1};    // p1=23
static constexpr STATE K3 = {1, 3, 10, 2};   // p2=19
static constexpr STATE K4 = {2, 2, 0, 4};    // p2=19
// Total period T: 64.7124 bits; T1: 1'779'648'563, T2: 16'983'563'040

struct gens {
    LFSR_1 gp1;
    LFSR_2 gp2;
    LFSR_3 gp3;
    LFSR_4 gp4;
    const u32 i1 = 0;
    const u32 j1 = 3;
    const u32 i2 = 2;
    const u32 j2 = 1;
    u32 x1;
    u32 x2;
    u32 x3;
    u32 x4;
public:
    constexpr gens(): gp1(K1)
        , gp2(K2)
        , gp3(K3)
        , gp4(K4) {}
    void seed(STATE st) {
        gp1.set_state(st);
        gp2.set_state(st);
        gp3.set_state(st);
        gp4.set_state(st);
        x1 = x2 = x3 = x4 = 1; // to exclude zero-loop
        for (int i=0; i<3*m; ++i) { // saturation
            gp1.next(x2);
            gp2.next(x1);
            gp3.next(x4);
            gp4.next(x3);
            x1 = gp1.get_cell(i1);
            x2 = gp2.get_cell(j1);
            x3 = gp3.get_cell(i2);
            x4 = gp4.get_cell(j2);
        }
    }
    std::array<u16, 4> next() {
        std::array<u16, 4> x;
        STATE mSt;
        for (int i=0; i<4; ++i) { // 16*4 bits
            gp1.next(x2);
            gp2.next(x1);
            gp3.next(x4);
            gp4.next(x3);
            x1 = gp1.get_cell(i1);
            x2 = gp2.get_cell(j1);
            x3 = gp3.get_cell(i2);
            x4 = gp4.get_cell(j2);
            mSt = gp1.get_state() ^ gp2.get_state();
            mSt ^= gp3.get_state() ^ gp4.get_state();
            mSt %= 16;
            x[i] <<= 4;
            x[i] |= mSt[0];
            x[i] <<= 4;
            x[i] |= mSt[3];
            x[i] <<= 4;
            x[i] |= mSt[1];
            x[i] <<= 4;
            x[i] |= mSt[2];
        }
        return x;
    }
};

}
