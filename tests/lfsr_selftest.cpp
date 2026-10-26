/**
 * @file lfsr_selftest.cpp
 * @brief Самопроверка генератора prng/lfsr.h.
 *
 * Эталонная модель (ground truth) - независимая скалярная реализация
 * генератора в поле GF(p^m), повторяющая референс lfsr.hpp
 * (\\wsl.localhost\Ubuntu-24.04\home\novikov\lfsr\lfsr.hpp):
 *
 *   next(input):  v = s[m-1];
 *                 s[i] = (s[i-1] + v*K[i])     % p,  i = m-1 .. 1;
 *                 s[0] = (input + v*K[0])      % p;
 *   back(input):  v = K[0]^(-1) * (s[0] - input) % p;
 *                 s[i] = (s[i+1] - v*K[i+1])   % p,  i = m-2 .. 0;
 *                 s[m-1] = v.
 *
 * Проверяется:
 *   A) next()     совпадает с моделью;
 *   B) back()     совпадает с моделью;
 *   C) next() + back() возвращают исходное состояние;
 *   D) длинный прогон: состояние не расходится с моделью;
 *   E) инвариант диапазона: все ячейки состояния < p;
 *   F) mult_by():  x^t * x^u = x^(t+u) (сверка с многократным next());
 *   G) square():  (x^t)^2    = x^(2t)  (сверка с многократным next());
 *   H) сдвоенный LFSR_paired_2x4: A, B, C, D.
 *
 * Коэффициент K[0] = p-1 выбирается специально: это максимально возможный
 * обратный коэффициент, он максимально нагружает модульное умножение в back().
 *
 * Код возврата: 0 - успех, 1 - есть расхождения.
 */
#include <cstdio>
#include <cstdint>
#include <random>

#include "lfsr.h"
#include "rng2.h"

namespace {

using lfsr8::u16;
using lfsr8::u32;
using lfsr8::u64;
using lfsr8::u16x8;

int g_checks = 0;
int g_failures = 0;

/**
 * @brief Учесть результат группы проверок.
 * @param label Метка случая (p, m).
 * @param what  Что проверялось.
 * @param bad   Число расхождений.
 * @param total Число проверок в группе.
 */
void record(const char* label, const char* what, int bad, int total)
{
    g_checks += total;
    g_failures += bad;
    if (bad == 0) {
        std::printf("  %-14s %-11s OK   (%6d)\n", label, what, total);
    } else {
        std::printf("  %-14s %-11s FAIL (%6d from %6d)\n", label, what, bad, total);
    }
}

/**
 * @brief Скалярная эталонная модель GF(p^m).
 */
template <int p, int m>
struct Model
{
    using SAMPLE = typename lfsr8::MType<m>::SAMPLE;
    using STATE  = typename lfsr8::MType<m>::STATE;

    static constexpr u64 P = static_cast<u64>(p);

    static u64 inverse(u64 a)
    {
        for (u64 i = 1; i < P; ++i) {
            if ((a * i) % P == 1) {
                return i;
            }
        }
        return 0;
    }

    static STATE next(STATE s, u64 input, const STATE& K)
    {
        const u64 v = s[m - 1];
        for (int i = m - 1; i > 0; --i) {
            s[i] = static_cast<SAMPLE>((static_cast<u64>(s[i - 1]) + v * K[i]) % P);
        }
        s[0] = static_cast<SAMPLE>((input + v * K[0]) % P);
        return s;
    }

    static STATE back(STATE s, u64 input, const STATE& K)
    {
        const u64 v = (inverse(K[0]) * ((static_cast<u64>(s[0]) + P - input % P) % P)) % P;
        for (int i = 0; i < m - 1; ++i) {
            s[i] = static_cast<SAMPLE>((static_cast<u64>(s[i + 1]) + P * P - v * K[i + 1]) % P);
        }
        s[m - 1] = static_cast<SAMPLE>(v);
        return s;
    }

    static bool equal(const STATE& a, const STATE& b)
    {
        for (int i = 0; i < m; ++i) {
            if (a[i] != b[i]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief Коэффициенты для теста: K[i] = i+1 (mod p), K[0] = p-1.
 */
template <int p, int m>
typename lfsr8::MType<m>::STATE make_K()
{
    using SAMPLE = typename lfsr8::MType<m>::SAMPLE;
    using STATE  = typename lfsr8::MType<m>::STATE;

    STATE K{};
    for (int i = 0; i < m; ++i) {
        K[i] = static_cast<SAMPLE>((i + 1) % p);
    }
    K[0] = static_cast<SAMPLE>(p - 1);
    return K;
}

/**
 * @brief Проверки A-E для LFSR<p, m>.
 */
template <int p, int m>
void test_general(const char* label, std::mt19937& rng, int iters)
{
    using SAMPLE = typename lfsr8::MType<m>::SAMPLE;
    using STATE  = typename lfsr8::MType<m>::STATE;
    using M      = Model<p, m>;

    const STATE K = make_K<p, m>();
    lfsr8::LFSR<p, m> gen(K);

    std::uniform_int_distribution<u32> dist(0, static_cast<u32>(p) - 1);

    int bad_next = 0;
    int bad_back = 0;
    int bad_round = 0;
    int bad_long = 0;
    int bad_range = 0;

    for (int it = 0; it < iters; ++it) {
        STATE s{};
        for (int i = 0; i < m; ++i) {
            s[i] = static_cast<SAMPLE>(dist(rng));
        }
        const u32 input = dist(rng);

        gen.set_state(s);
        gen.next(input);
        if (!M::equal(gen.get_state(), M::next(s, input, K))) {
            ++bad_next;
        }

        gen.set_state(s);
        gen.back(input);
        if (!M::equal(gen.get_state(), M::back(s, input, K))) {
            ++bad_back;
        }

        gen.set_state(s);
        gen.next(input);
        gen.back(input);
        if (!M::equal(gen.get_state(), s)) {
            ++bad_round;
        }

        gen.set_state(s);
        STATE ref = s;
        bool range_ok = true;
        for (int k = 0; k < 32; ++k) {
            const u32 in = dist(rng);
            gen.next(in);
            ref = M::next(ref, in, K);
            const STATE st = gen.get_state();
            for (int i = 0; i < m; ++i) {
                if (st[i] >= static_cast<SAMPLE>(p)) {
                    range_ok = false;
                    break;
                }
            }
            if (!M::equal(st, ref)) {
                ++bad_long;
                break;
            }
        }
        if (!range_ok) {
            ++bad_range;
        }
    }

    record(label, "next()", bad_next, iters);
    record(label, "back()", bad_back, iters);
    record(label, "round-trip", bad_round, iters);
    record(label, "long-run", bad_long, iters);
    record(label, "range", bad_range, iters);
}

/**
 * @brief Проверки F-G: mult_by() и square() через многократный next().
 */
template <int p, int m>
void test_algebra(const char* label, int t, int u)
{
    using STATE = typename lfsr8::MType<m>::STATE;
    using M     = Model<p, m>;

    const STATE K = make_K<p, m>();

    const auto x_pow = [&](int e) {
        lfsr8::LFSR<p, m> g(K);
        g.set_unit_state();
        for (int i = 0; i < e; ++i) {
            g.next();
        }
        return g.get_state();
    };

    const STATE xt  = x_pow(t);
    const STATE xu  = x_pow(u);
    const STATE xtu = x_pow(t + u);
    const STATE x2t = x_pow(2 * t);

    lfsr8::LFSR<p, m> gm(K);
    gm.set_state(xt);
    gm.mult_by(xu);
    const bool ok_mult = M::equal(gm.get_state(), xtu);

    lfsr8::LFSR<p, m> gs(K);
    gs.set_state(xt);
    gs.square();
    const bool ok_square = M::equal(gs.get_state(), x2t);

    record(label, "mult_by()", ok_mult ? 0 : 1, 1);
    record(label, "square()", ok_square ? 0 : 1, 1);
}

u64 inverse_mod(u64 a, u64 p)
{
    for (u64 i = 1; i < p; ++i) {
        if ((a * i) % p == 1) {
            return i;
        }
    }
    return 0;
}

bool equal8(const u16x8& a, const u16x8& b)
{
    for (int i = 0; i < 8; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

/** Модель прямого такта сдвоенного генератора. */
u16x8 paired_next_model(u16x8 e, const u16x8& K, u64 P, u16 a, u16 b)
{
    const u16 v3 = e[3];
    const u16 v7 = e[7];
    for (int i = 7; i > 4; --i) {
        e[i]     = static_cast<u16>((static_cast<u64>(e[i - 1]) + static_cast<u64>(v7) * K[i]) % P);
        e[i - 4] = static_cast<u16>((static_cast<u64>(e[i - 5]) + static_cast<u64>(v3) * K[i - 4]) % P);
    }
    e[0] = static_cast<u16>((static_cast<u64>(a) + static_cast<u64>(v3) * K[0]) % P);
    e[4] = static_cast<u16>((static_cast<u64>(b) + static_cast<u64>(v7) * K[4]) % P);
    return e;
}

/** Модель обратного такта сдвоенного генератора. */
u16x8 paired_back_model(u16x8 e, const u16x8& K, u64 P, u64 inv0, u64 inv4, u16 a, u16 b)
{
    const u64 v1 = (inv0 * ((static_cast<u64>(e[0]) + P - a % P) % P)) % P;
    const u64 v2 = (inv4 * ((static_cast<u64>(e[4]) + P - b % P) % P)) % P;
    for (int i = 0; i < 3; ++i) {
        e[i]     = static_cast<u16>((static_cast<u64>(e[i + 1]) + P * P - v1 * K[i + 1]) % P);
        e[i + 4] = static_cast<u16>((static_cast<u64>(e[i + 5]) + P * P - v2 * K[i + 5]) % P);
    }
    e[3] = static_cast<u16>(v1);
    e[7] = static_cast<u16>(v2);
    return e;
}

/**
 * @brief Проверки A-E для LFSR_paired_2x4<p>.
 */
template <int p>
void test_paired(const char* label, std::mt19937& rng, int iters)
{
    u16x8 K{};
    for (int i = 0; i < 8; ++i) {
        K[i] = static_cast<u16>((i + 1) % p);
    }
    K[0] = static_cast<u16>(p - 1);
    K[4] = static_cast<u16>(p - 1);

    const u64 P = p;
    const u64 inv0 = inverse_mod(K[0], P);
    const u64 inv4 = inverse_mod(K[4], P);

    lfsr8::LFSR_paired_2x4<p> gen(K);
    std::uniform_int_distribution<u32> dist(0, static_cast<u32>(p) - 1);

    int bad_next = 0;
    int bad_back = 0;
    int bad_round = 0;
    int bad_range = 0;

    for (int it = 0; it < iters; ++it) {
        u16x8 s{};
        for (int i = 0; i < 8; ++i) {
            s[i] = static_cast<u16>(dist(rng));
        }
        const u16 in1 = static_cast<u16>(dist(rng));
        const u16 in2 = static_cast<u16>(dist(rng));

        gen.set_state(s);
        gen.next(in1, in2);
        if (!equal8(gen.get_state(), paired_next_model(s, K, P, in1, in2))) {
            ++bad_next;
        }

        gen.set_state(s);
        gen.back(in1, in2);
        if (!equal8(gen.get_state(), paired_back_model(s, K, P, inv0, inv4, in1, in2))) {
            ++bad_back;
        }

        gen.set_state(s);
        gen.next(in1, in2);
        gen.back(in1, in2);
        if (!equal8(gen.get_state(), s)) {
            ++bad_round;
        }

        gen.set_state(s);
        u16x8 ref = s;
        bool range_ok = true;
        for (int k = 0; k < 32; ++k) {
            const u16 a = static_cast<u16>(dist(rng));
            const u16 b = static_cast<u16>(dist(rng));
            gen.next(a, b);
            ref = paired_next_model(ref, K, P, a, b);
            const u16x8 st = gen.get_state();
            for (int i = 0; i < 8; ++i) {
                if (st[i] >= static_cast<u16>(p)) {
                    range_ok = false;
                    break;
                }
            }
            if (!equal8(st, ref)) {
                break;
            }
        }
        if (!range_ok) {
            ++bad_range;
        }
    }

    record(label, "next()", bad_next, iters);
    record(label, "back()", bad_back, iters);
    record(label, "round-trip", bad_round, iters);
    record(label, "range", bad_range, iters);
}

/**
 * @brief Эталонный генератор rng2: независимая реализация упаковки.
 * @details Перенос next_u64() из
 * \\wsl.localhost\Ubuntu-24.04\home\novikov\lfsr\random_gen_2.hpp,
 * но на prng/lfsr.h (проверенном выше). Сам файл эталона заморожен и
 * содержит старую упаковку (XOR старших битов в бит 0), поэтому здесь
 * повторена исправленная упаковка prng/rng2.h (XOR в бит 3 полубайта).
 */
class RefGens2
{
    rng2::LFSR_1 gp1{rng2::K1};
    rng2::LFSR_2 gp2{rng2::K2};
    rng2::LFSR_3 gp3{rng2::K3};
    rng2::LFSR_4 gp4{rng2::K4};
    u32 x1{}, x2{}, x3{}, x4{};

public:
    void seed(rng2::STATE st)
    {
        gp1.set_state(st);
        gp2.set_state(st);
        gp3.set_state(st);
        gp4.set_state(st);
        x1 = x2 = x3 = x4 = 1;
        for (int i = 0; i < 3 * rng2::m; ++i) {
            gp1.next(x2);
            gp2.next(x1);
            gp3.next(x4);
            gp4.next(x3);
            x1 = gp1.get_cell(0);
            x2 = gp2.get_cell(3);
            x3 = gp3.get_cell(2);
            x4 = gp4.get_cell(1);
        }
    }

    u64 next_u64()
    {
        using namespace rng2; // operator^=, operator/, operator%= для STATE
        u64 x = 0;
        for (int i = 0; i < 4; ++i) { // 16*4 bits
            gp1.next(x2);
            gp2.next(x1);
            gp3.next(x4);
            gp4.next(x3);
            x1 = gp1.get_cell(0);
            x2 = gp2.get_cell(3);
            x3 = gp3.get_cell(2);
            x4 = gp4.get_cell(1);
            auto mSt = gp1.get_state();
            mSt ^= gp2.get_state();
            mSt ^= gp3.get_state();
            mSt ^= gp4.get_state();
            const auto high_bits = mSt / 16;
            mSt %= 16;
            x <<= 4;
            x |= mSt[0];
            x ^= high_bits[1] << 3;
            x <<= 4;
            x |= mSt[2];
            x ^= high_bits[0] << 3;
            x <<= 4;
            x |= mSt[3];
            x ^= high_bits[2] << 3;
            x <<= 4;
            x |= mSt[1];
            x ^= high_bits[3] << 3;
        }
        return x;
    }
};

/**
 * @brief Сверка rng2::gens::next() с эталонным next_u64().
 * @details Четыре u16 от rng2 склеиваются в тот же u64: x[0] - старшее слово.
 * Начальное состояние берётся из rng (все ячейки < p1 = 23).
 */
void test_rng2(const char* label, std::mt19937& rng, int iters)
{
    std::uniform_int_distribution<u32> dist(0, rng2::p1 - 1);

    rng2::STATE st{};
    for (int i = 0; i < rng2::m; ++i) {
        st[i] = dist(rng);
    }

    rng2::gens gen;
    gen.seed(st);
    RefGens2 ref;
    ref.seed(st);

    int bad = 0;
    for (int i = 0; i < iters; ++i) {
        const auto x = gen.next();
        const u64 got = (static_cast<u64>(x[0]) << 48) | (static_cast<u64>(x[1]) << 32)
                      | (static_cast<u64>(x[2]) << 16) | static_cast<u64>(x[3]);
        if (got != ref.next_u64()) {
            ++bad;
        }
    }
    record(label, "next_u64", bad, iters);
}

} // namespace

int main()
{
    std::printf("LFSR self-test: prng/lfsr.h vs scalar model\n\n");

    std::mt19937 rng(20261019u);
    const int iters = 2000;

    std::printf("[general] LFSR<p, m>, K[0] = p-1\n");
    test_general<2, 1>("p=2 m=1", rng, iters);
    test_general<2, 4>("p=2 m=4", rng, iters);
    test_general<3, 2>("p=3 m=2", rng, iters);
    test_general<19, 4>("p=19 m=4", rng, iters);
    test_general<23, 4>("p=23 m=4", rng, iters);
    test_general<181, 5>("p=181 m=5", rng, iters);
    test_general<191, 5>("p=191 m=5", rng, iters);
    test_general<251, 5>("p=251 m=5", rng, iters);
    test_general<251, 8>("p=251 m=8", rng, iters);
    test_general<46337, 4>("p=46337 m=4", rng, iters);
    test_general<46349, 4>("p=46349 m=4", rng, iters);
    test_general<65521, 2>("p=65521 m=2", rng, iters);
    test_general<65521, 4>("p=65521 m=4", rng, iters);

    std::printf("\n[algebra] x^t * x^u = x^(t+u), (x^t)^2 = x^(2t)\n");
    test_algebra<2, 4>("p=2 m=4", 1000, 700);
    test_algebra<23, 4>("p=23 m=4", 1000, 700);
    test_algebra<251, 8>("p=251 m=8", 1000, 700);
    test_algebra<65521, 4>("p=65521 m=4", 1000, 700);

    std::printf("\n[paired] LFSR_paired_2x4<p>, K[0] = K[4] = p-1\n");
    test_paired<2>("p=2", rng, iters);
    test_paired<3>("p=3", rng, iters);
    test_paired<23>("p=23", rng, iters);
    test_paired<251>("p=251", rng, iters);

    std::printf("\n[rng2] gens::next() vs RefGens2 (next_u64)\n");
    test_rng2("seed#1", rng, iters);
    test_rng2("seed#2", rng, iters);
    test_rng2("seed#3", rng, 10 * iters);
    test_rng2("seed#4", rng, iters);

    std::printf("\nchecks: %d, failures: %d -> %s\n",
                g_checks, g_failures, g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
