/**
 * @file rfft_selftest.cpp
 * @brief Самопроверка БПФ rfft/rfft.cpp.
 *
 * Эталон - независимая реализация БПФ в double (итеративный radix-2 с
 * бит-реверсной перестановкой), не использующая код rfft. Сам эталон
 * сверяется с наивным ДПФ (O(N^2)) на малых N, а также был сверен с
 * эталонным проектом c:\Users\naw19\qt_projects\fft (FFTIterativeAoS, double):
 * расхождение не более 1.5e-13 при N = 1024.
 *
 * Проверяется:
 *   A) документированный в rfft.h пример для N = 4;
 *   B) импульс в N/2 даёт спектр (-1)^k;
 *   C) прямое и обратное преобразование против эталона: реальный и
 *      комплексный вход, N = 2..16384, RfftSSE3 и Rfftx86, режимы
 *      CONJUGATE и RADIX2 (у Rfftx86 CONJUGATE должен давать radix-2);
 *   D) round-trip: ifft(fft(x))/N == x;
 *   E) повторные вызовы, смена размера и режима на одном объекте.
 *
 * Код возврата: 0 - успех, 1 - есть расхождения.
 */
#include <pmmintrin.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "rfft.h"

namespace {

using C64 = std::complex<double>;

constexpr double kPi = 3.1415926535897932384626433832795;

/** Допустимое относительное расхождение (вход/эталон в float32). */
constexpr double kTolRel = 2e-5;

/** Допустимое абсолютное расхождение для спектра импульса. */
constexpr double kTolAbs = 1e-3;

int g_checks = 0;
int g_failures = 0;

/**
 * @brief Буфер, выровненный на 16 байт, как того требует rfft.
 */
class AlignedBuf
{
public:
    explicit AlignedBuf(size_t n)
        : m_ptr(static_cast<data_t*>(_mm_malloc(n * sizeof(data_t), 16)))
    {
    }
    ~AlignedBuf() { _mm_free(m_ptr); }
    AlignedBuf(const AlignedBuf&) = delete;
    AlignedBuf& operator=(const AlignedBuf&) = delete;

    data_t* data() { return m_ptr; }
    data_t& operator[](size_t i) { return m_ptr[i]; }
    const data_t& operator[](size_t i) const { return m_ptr[i]; }

private:
    data_t* m_ptr;
};

/** @brief log2 для точной степени двойки. */
size_t ilog2(size_t n)
{
    size_t k = 0;
    while ((size_t(1) << k) < n) {
        ++k;
    }
    return k;
}

/**
 * @brief Наивное ДПФ в double - используется только для проверки эталона.
 * @param inverse true - знак показателя положительный (без нормировки 1/N).
 */
std::vector<C64> naive_dft(const std::vector<C64>& x, bool inverse)
{
    const size_t n = x.size();
    std::vector<C64> y(n);
    const double sign = inverse ? 1.0 : -1.0;
    for (size_t k = 0; k < n; ++k) {
        C64 acc(0.0, 0.0);
        for (size_t j = 0; j < n; ++j) {
            const double ang = sign * 2.0 * kPi * double((j * k) % n) / double(n);
            acc += x[j] * C64(std::cos(ang), std::sin(ang));
        }
        y[k] = acc;
    }
    return y;
}

/**
 * @brief Эталонное БПФ в double, итеративный radix-2, без нормировки.
 * @param a   Вход/выход, размер - степень двойки.
 * @param inverse true - обратное преобразование (знак +).
 */
void ref_fft(std::vector<C64>& a, bool inverse)
{
    const size_t n = a.size();
    if (n <= 1) {
        return;
    }

    // Бит-реверсная перестановка.
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    const double sign = inverse ? 1.0 : -1.0;
    for (size_t len = 2; len <= n; len <<= 1) {
        const size_t half = len >> 1;
        for (size_t start = 0; start < n; start += len) {
            for (size_t j = 0; j < half; ++j) {
                // Тригонометрия на каждую бабочку: точнее, чем накопление w.
                const double ang = sign * 2.0 * kPi * double(j) / double(len);
                const C64 w(std::cos(ang), std::sin(ang));
                const C64 u = a[start + j];
                const C64 v = a[start + j + half] * w;
                a[start + j] = u + v;
                a[start + j + half] = u - v;
            }
        }
    }
}

/**
 * @brief Учесть результат группы проверок.
 * @param label Метка случая (реализация/режим).
 * @param what  Что проверялось.
 * @param bad   Число расхождений.
 * @param total Число проверок в группе.
 * @param max_rel Максимальное относительное расхождение.
 */
void record(const char* label, const char* what, int bad, int total, double max_rel)
{
    g_checks += total;
    g_failures += bad;
    if (bad == 0) {
        std::printf("  %-16s %-11s OK   (%6d, max %.1e)\n", label, what, total, max_rel);
    } else {
        std::printf("  %-16s %-11s FAIL (%6d from %6d, max %.1e)\n",
                    label, what, bad, total, max_rel);
    }
}

/**
 * @brief Относительное расхождение результата rfft с эталоном.
 */
double max_rel_err(const data_t* got, const std::vector<C64>& ref)
{
    double max_err = 0.0;
    double max_ref = 0.0;
    for (size_t k = 0; k < ref.size(); ++k) {
        const C64 g(got[k].real(), got[k].imag());
        if (!std::isfinite(g.real()) || !std::isfinite(g.imag())) {
            return INFINITY;
        }
        max_err = std::max(max_err, std::abs(g - ref[k]));
        max_ref = std::max(max_ref, std::abs(ref[k]));
    }
    return max_err / std::max(max_ref, 1e-30);
}

/** @brief Набор эталонных спектров для одного размера. */
struct SizeRefs
{
    size_t n = 0;
    std::vector<C64> real_in, cplx_in;   // входные сигналы
    std::vector<C64> real_fwd, real_inv; // эталонные спектры для real_in
    std::vector<C64> cplx_fwd, cplx_inv; // эталонные спектры для cplx_in
};

/**
 * @brief Проверки A-B: документированный пример и спектр импульса.
 */
void test_known_cases(Rfft* fft, const char* label)
{
    // A) Пример из rfft.h: v = (0, 1+i, 2+2i, 3+3i).
    {
        const data_t in[4] = {{0.f, 0.f}, {1.f, 1.f}, {2.f, 2.f}, {3.f, 3.f}};
        const data_t expected_fwd[4] = {{6.f, 6.f}, {-4.f, 0.f}, {-2.f, -2.f}, {0.f, -4.f}};
        const data_t expected_inv[4] = {{6.f, 6.f}, {0.f, -4.f}, {-2.f, -2.f}, {-4.f, 0.f}};

        AlignedBuf in_buf(4);
        AlignedBuf out_buf(4);
        for (int i = 0; i < 4; ++i) {
            in_buf[i] = in[i];
        }

        fft->setLogSize(2);
        fft->c2cfft(in_buf.data(), out_buf.data());
        int bad = 0;
        for (int i = 0; i < 4; ++i) {
            if (std::abs(out_buf[i] - expected_fwd[i]) > 1e-4f) {
                ++bad;
            }
        }
        fft->c2cifft(in_buf.data(), out_buf.data());
        for (int i = 0; i < 4; ++i) {
            if (std::abs(out_buf[i] - expected_inv[i]) > 1e-4f) {
                ++bad;
            }
        }
        record(label, "example N=4", bad, 8, 0.0);
    }

    // B) Импульс в N/2: X[k] = (-1)^k.
    for (size_t n : {64u, 1024u, 8192u}) {
        AlignedBuf in_buf(n);
        AlignedBuf out_buf(n);
        for (size_t i = 0; i < n; ++i) {
            in_buf[i] = data_t(0.f, 0.f);
        }
        in_buf[n / 2] = data_t(1.f, 0.f);

        fft->setLogSize(ilog2(n));
        fft->c2cfft(in_buf.data(), out_buf.data());

        int bad = 0;
        double max_err = 0.0;
        for (size_t k = 0; k < n; ++k) {
            const data_t expected((k % 2 == 0) ? 1.f : -1.f, 0.f);
            const double err = std::abs(out_buf[k] - expected);
            max_err = std::max(max_err, err);
            if (err > kTolAbs) {
                ++bad;
            }
        }
        record(label, "impulse", bad, int(n), max_err);
    }
}

/**
 * @brief Проверки C-D для одной реализации и одного режима.
 */
void test_transform(const char* impl, int mode, const std::vector<SizeRefs>& refs)
{
    Rfft* fft = nullptr;
    if (std::string(impl) == "SSE3") {
        fft = new RfftSSE3;
    } else {
        fft = new Rfftx86;
    }
    fft->setMode(mode);

    const char* mode_name = (mode == Rfft::CONJUGATE) ? "CONJUGATE" : "RADIX2";
    std::string label = std::string(impl) + "/" + mode_name;

    int bad_fwd_real = 0, bad_inv_real = 0, bad_fwd_cplx = 0, bad_inv_cplx = 0, bad_rt = 0;
    double rel_fwd_real = 0.0, rel_inv_real = 0.0, rel_fwd_cplx = 0.0, rel_inv_cplx = 0.0;
    double rel_rt = 0.0;

    for (const SizeRefs& r : refs) {
        const size_t n = r.n;
        fft->setLogSize(ilog2(n));

        // C) прямое/обратное против эталона.
        AlignedBuf in_buf(n);
        AlignedBuf out_buf(n);
        for (size_t i = 0; i < n; ++i) {
            in_buf[i] = data_t(float(r.real_in[i].real()), float(r.real_in[i].imag()));
        }
        fft->c2cfft(in_buf.data(), out_buf.data());
        const double e1 = max_rel_err(out_buf.data(), r.real_fwd);
        rel_fwd_real = std::max(rel_fwd_real, e1);
        if (!(e1 <= kTolRel)) {
            ++bad_fwd_real;
            std::printf("    fwd real N=%zu rel=%.3e\n", n, e1);
        }

        fft->c2cifft(in_buf.data(), out_buf.data());
        const double e2 = max_rel_err(out_buf.data(), r.real_inv);
        rel_inv_real = std::max(rel_inv_real, e2);
        if (!(e2 <= kTolRel)) {
            ++bad_inv_real;
            std::printf("    inv real N=%zu rel=%.3e\n", n, e2);
        }

        for (size_t i = 0; i < n; ++i) {
            in_buf[i] = data_t(float(r.cplx_in[i].real()), float(r.cplx_in[i].imag()));
        }
        fft->c2cfft(in_buf.data(), out_buf.data());
        const double e3 = max_rel_err(out_buf.data(), r.cplx_fwd);
        rel_fwd_cplx = std::max(rel_fwd_cplx, e3);
        if (!(e3 <= kTolRel)) {
            ++bad_fwd_cplx;
            std::printf("    fwd cplx N=%zu rel=%.3e\n", n, e3);
        }

        fft->c2cifft(in_buf.data(), out_buf.data());
        const double e4 = max_rel_err(out_buf.data(), r.cplx_inv);
        rel_inv_cplx = std::max(rel_inv_cplx, e4);
        if (!(e4 <= kTolRel)) {
            ++bad_inv_cplx;
            std::printf("    inv cplx N=%zu rel=%.3e\n", n, e4);
        }

        // D) round-trip: ifft(fft(x)) / N == x.
        if (n >= 8) {
            AlignedBuf tmp(n);
            AlignedBuf back(n);
            fft->c2cfft(in_buf.data(), tmp.data());
            fft->c2cifft(tmp.data(), back.data());
            double max_err = 0.0;
            double max_abs = 0.0;
            for (size_t i = 0; i < n; ++i) {
                max_err = std::max(max_err, double(std::abs(back[i] / float(n) - in_buf[i])));
                max_abs = std::max(max_abs, double(std::abs(in_buf[i])));
            }
            const double e5 = max_err / std::max(max_abs, 1e-30);
            rel_rt = std::max(rel_rt, e5);
            if (!(e5 <= kTolRel)) {
                ++bad_rt;
                std::printf("    round-trip N=%zu rel=%.3e\n", n, e5);
            }
        }
    }

    const int sizes = int(refs.size());
    record(label.c_str(), "fwd real", bad_fwd_real, sizes, rel_fwd_real);
    record(label.c_str(), "inv real", bad_inv_real, sizes, rel_inv_real);
    record(label.c_str(), "fwd cplx", bad_fwd_cplx, sizes, rel_fwd_cplx);
    record(label.c_str(), "inv cplx", bad_inv_cplx, sizes, rel_inv_cplx);
    record(label.c_str(), "round-trip", bad_rt, sizes - 2, rel_rt);

    delete fft;
}

/**
 * @brief Проверка E: смена размера и режима на живом объекте.
 */
void test_state_switching(const std::vector<SizeRefs>& refs)
{
    const SizeRefs& r = refs.back(); // N = 16384
    const size_t n = r.n;
    AlignedBuf in_buf(n);
    AlignedBuf out_buf(n);
    for (size_t i = 0; i < n; ++i) {
        in_buf[i] = data_t(float(r.real_in[i].real()), 0.f);
    }

    int bad = 0;
    double max_rel = 0.0;
    const auto check = [&](Rfft* fft, const char* what) {
        fft->c2cfft(in_buf.data(), out_buf.data());
        const double e = max_rel_err(out_buf.data(), r.real_fwd);
        max_rel = std::max(max_rel, e);
        if (!(e <= kTolRel)) {
            ++bad;
            std::printf("    %s rel=%.3e\n", what, e);
        }
    };

    {
        // Смена режима и размера: 16384 -> 16 -> 16384.
        RfftSSE3 fft;
        fft.setLogSize(ilog2(n));
        fft.setMode(Rfft::RADIX2);
        check(&fft, "SSE3 RADIX2");
        fft.setMode(Rfft::CONJUGATE);
        check(&fft, "SSE3 CONJUGATE");
        fft.setLogSize(4);
        fft.setLogSize(ilog2(n));
        check(&fft, "SSE3 CONJUGATE after resize");
    }
    {
        // Rfftx86: режим CONJUGATE не поддержан и должен давать radix-2.
        Rfftx86 fft;
        fft.setLogSize(ilog2(n));
        fft.setMode(Rfft::CONJUGATE);
        check(&fft, "x86 CONJUGATE->RADIX2");
        fft.setMode(Rfft::RADIX2);
        check(&fft, "x86 RADIX2");
    }

    record("state", "switch", bad, 5, max_rel);
}

} // namespace

int main()
{
    std::printf("RFFT self-test: rfft/rfft.cpp vs double reference\n\n");

    // 0. Проверка самого эталона против наивного ДПФ.
    {
        std::mt19937 rng(20261025u);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        int bad = 0;
        double max_err = 0.0;
        int total = 0;
        for (size_t n : {8u, 16u, 64u, 256u}) {
            std::vector<C64> x(n);
            for (C64& v : x) {
                v = C64(dist(rng), dist(rng));
            }
            for (bool inverse : {false, true}) {
                std::vector<C64> a = x;
                ref_fft(a, inverse);
                const auto b = naive_dft(x, inverse);
                double err = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    err = std::max(err, std::abs(a[k] - b[k]));
                }
                max_err = std::max(max_err, err);
                ++total;
                if (err > 1e-10) {
                    ++bad;
                }
            }
        }
        record("reference", "vs DFT", bad, total, max_err);
    }

    // Подготовка эталонных спектров для всех размеров.
    std::vector<SizeRefs> refs;
    {
        std::mt19937 rng(12345u);
        std::uniform_real_distribution<double> dist(-4000.0, 4000.0);
        for (int lg = 1; lg <= 14; ++lg) {
            SizeRefs r;
            r.n = size_t(1) << lg;
            std::vector<C64> real_in(r.n), cplx_in(r.n);
            for (size_t i = 0; i < r.n; ++i) {
                real_in[i] = C64(dist(rng), 0.0);
                cplx_in[i] = C64(dist(rng), dist(rng));
            }
            r.real_in = real_in;
            r.cplx_in = cplx_in;
            r.real_fwd = real_in;
            ref_fft(r.real_fwd, false);
            r.real_inv = real_in;
            ref_fft(r.real_inv, true);
            r.cplx_fwd = cplx_in;
            ref_fft(r.cplx_fwd, false);
            r.cplx_inv = cplx_in;
            ref_fft(r.cplx_inv, true);
            refs.push_back(std::move(r));
        }
    }

    // A-B. Известные случаи.
    std::printf("[known] документированный пример и импульс\n");
    {
        RfftSSE3 fft;
        test_known_cases(&fft, "SSE3");
        Rfftx86 fft_x86;
        test_known_cases(&fft_x86, "x86");
    }

    // C-D. Сверка с эталоном.
    std::printf("\n[transform] N = 2..16384, реальный и комплексный вход\n");
    for (const char* impl : {"SSE3", "x86"}) {
        for (int mode : {int(Rfft::CONJUGATE), int(Rfft::RADIX2)}) {
            test_transform(impl, mode, refs);
        }
    }

    // E. Смена размера и режима.
    std::printf("\n[state] смена размера и режима\n");
    test_state_switching(refs);

    std::printf("\nchecks: %d, failures: %d -> %s\n",
                g_checks, g_failures, g_failures ? "FAILED" : "PASSED");
    return g_failures ? 1 : 0;
}
