#pragma once

/**
 * @author Новиков А.В.
 *
 * Генератор LFSR в поле GF(p^m) с числом ячеек m = [1, 8] и простым модулем p = [2, 256*256).
 * Реализован код для SSE4.1 архитектуры x86_64.
 * Реализованы:
 *  - Генератор общего назначения, m = [1, 8], p = [2, 256) для m = [5, 8] и p = [2, 256*256) для m = [1, 4].
 *  - Сдвоенный генератор с фиксированным m = 4, p = [2, 256). Фактически, это два независимых генератора,
 *  реализованных на одном регистре 128-бит в коде SSE4.1.
*/

#include <cstdint>
#include <cassert>
#include <array>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
#define USE_SSE
#endif

#ifdef USE_SSE
#include <immintrin.h>
#include <smmintrin.h>
#endif


namespace lfsr8 {
using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using u16x8 = std::array<u16, 8>;
using u32x4 = std::array<u32, 4>;

#ifdef USE_SSE
/**
 * @brief Векторное взятие остатка x % p для восьми 16-битных ячеек.
 * @details Метод Барретта (m = floor(2^16 / p), q = (x*m) >> 16, res = x - q*p),
 * как в проекте AllPass (utils/lfsr/lfsr.h). Корректно при 0 <= x < 2^16 и p < 256.
 * Применяется только к произведениям, у которых оба сомножителя уже приведены
 * по модулю p, поэтому x < p^2 < 2^16 и усечения нет.
 */
template <int p>
inline __m128i simd_mod_p16(__m128i x) {
    static_assert(p > 1);
    static_assert(p < 256);
    const __m128i m = _mm_set1_epi16(static_cast<short>((1u << 16) / p));
    const __m128i p_vec = _mm_set1_epi16(static_cast<short>(p));
    const __m128i q = _mm_mulhi_epu16(x, m);
    __m128i res = _mm_sub_epi16(x, _mm_mullo_epi16(q, p_vec));
    // Остаток после Барретта лежит в [0, 2p), достаточно одной коррекции.
    const __m128i ge = _mm_cmpgt_epi16(res, _mm_set1_epi16(static_cast<short>(p - 1)));
    return _mm_sub_epi16(res, _mm_and_si128(ge, p_vec));
}
#endif

// https://stackoverflow.com/questions/24053582/change-member-typedef-depending-on-template-parameter
template <int m>
class MType {
public:
    typedef typename std::conditional<(m <= 4), u32, u16>::type SAMPLE;
    typedef typename std::conditional<(m <= 4), u32x4, u16x8>::type STATE;
};

/**
 * @brief Генератор LFSR общего назначения в поле GF(p^m).
 * p должно быть простым числом в итервале:
 *   [2, 256) для длин регистра (4, 8].
 *   [2, 256*256) для длин регистра [1, 4].
 * m - длина регистра [1, 8].
 */
template <int p, int m>
class LFSR {
    using STATE = typename MType<m>::STATE;
    using SAMPLE = typename MType<m>::SAMPLE;
public:
    constexpr LFSR(STATE K): m_K(K) {
        static_assert(m <= 8);
        static_assert(m > 0);
        if constexpr (m > 4) {
            static_assert(p < 256);
        } else {
            static_assert(p < 256*256);
        }
        static_assert(p > 1);
        m_calculate_inverse_of_K();
    };

    void set_state(STATE st) {
        m_state = st;
    }

    void set_unit_state() {
        m_state = {1};
    }

    void set_K(STATE K) {
        m_K = K;
        m_calculate_inverse_of_K();
    }

    /**
     * @brief Сделать шаг вперед (один такт генератора).
     * @param input Входной символ (по модулю p), который подается
     * на вход генератора.
     */
    void next(SAMPLE input=0) {
#ifdef USE_SSE
        if constexpr (m > 4) {
            __m128i a = _mm_set1_epi16(m_state[m-1]);
            __m128i b = _mm_load_si128((const __m128i*)&m_K[0]);
            __m128i tmp = _mm_mullo_epi16(a, b);
            __m128i c = _mm_load_si128((const __m128i*)&m_state[0]);
            c = _mm_slli_si128(c, 2);
            __m128i mask = _mm_slli_si128(_mm_set1_epi16(-1), 2);
            __m128i inp = _mm_andnot_si128( mask, _mm_set1_epi16(input) );
            c = _mm_add_epi16(c, tmp);
            c = _mm_add_epi16(c, inp);
            _mm_store_si128((__m128i*)&m_state[0], c);
            for (int i=0; i<m; ++i) {
                m_state[i] %= static_cast<SAMPLE>(p);
            }
        } else {
            __m128i a = _mm_set1_epi32(m_state[m-1]);
            __m128i b = _mm_load_si128((const __m128i*)&m_K[0]);
            __m128i tmp = _mm_mullo_epi32(a, b);
            __m128i c = _mm_load_si128((const __m128i*)&m_state[0]);
            c = _mm_slli_si128(c, 4);
            __m128i mask = _mm_slli_si128(_mm_set1_epi32(-1), 4);
            __m128i inp = _mm_andnot_si128( mask, _mm_set1_epi32(input) );
            c = _mm_add_epi32(c, tmp);
            c = _mm_add_epi32(c, inp);
            _mm_store_si128((__m128i*)&m_state[0], c);
            for (int i=0; i<m; ++i) {
                m_state[i] %= static_cast<SAMPLE>(p);
            }
        }
#else // Для процессора общего назначения.
        const SAMPLE m_v = m_state[m-1];
        for (int i=m-1; i>0; i--) {
            m_state[i] = (m_state[i-1] + m_v*m_K[i]) % static_cast<SAMPLE>(p);
        }
        m_state[0] = (input + m_v*m_K[0]) % static_cast<SAMPLE>(p);
#endif
    }

    /**
     * @brief Сделать шаг назад (один такт генератора). Обратно к next(input).
     * @param inp Входной символ (по модулю p), который подается
     * на вход генератора.
     * @details Реализация в стиле AllPass (utils/lfsr/lfsr.h): разность
     * (m_state[0] - inp) приводится по модулю p ДО умножения на m_inv_K[0],
     * поэтому произведение не превышает p^2 < 2^16 (ячейки u16, p < 256) и
     * не усекается _mm_mullo_epi16. Раньше умножалось (state - inp + p) < 2p,
     * произведение достигало 2*p^2 > 2^16 и молча усекалось, из-за чего
     * коэффициент m_v был неверным при p > 181.
     */
    void back(SAMPLE inp=0) {
#ifdef USE_SSE
        if constexpr (m > 4) {
            const __m128i m0 = _mm_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0);
            const __m128i p_vec = _mm_set1_epi16(static_cast<short>(p));
            const __m128i st = _mm_load_si128((const __m128i*)&m_state[0]);

            // d[0] = (state[0] - inp) mod p: т.к. state[0] < p и inp < p,
            // вычитание по модулю 2^16 само достраивает +p.
            __m128i d = _mm_and_si128(m0, st);
            d = _mm_sub_epi16(d, _mm_and_si128(m0, _mm_set1_epi16(static_cast<short>(inp))));
            d = simd_mod_p16<p>(_mm_add_epi16(d, _mm_and_si128(m0, p_vec)));

            // v = d * inv_K[0] mod p (сомножители < p, произведение < p^2 < 2^16).
            __m128i v = simd_mod_p16<p>(_mm_mullo_epi16(d, _mm_load_si128((const __m128i*)&m_inv_K[0])));
            v = _mm_shuffle_epi8(v, _mm_setr_epi8(0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1, 0,1));

            // state[i] = (state[i+1] - v * K[i+1]) mod p, state[m-1] = v.
            const __m128i next_st = _mm_srli_si128(st, 2);
            const __m128i next_k = _mm_srli_si128(_mm_load_si128((const __m128i*)&m_K[0]), 2);
            const __m128i prod = simd_mod_p16<p>(_mm_mullo_epi16(v, next_k));
            __m128i res = _mm_sub_epi16(next_st, prod);
            res = _mm_add_epi16(res, _mm_and_si128(_mm_cmpgt_epi16(_mm_setzero_si128(), res), p_vec));
            res = _mm_blend_epi16(res, v, 1 << (m - 1));
            _mm_store_si128((__m128i*)&m_state[0], res);
        } else {
            const __m128i m0 = _mm_setr_epi32(-1, 0, 0, 0);
            const __m128i p_vec = _mm_set1_epi32(static_cast<int>(p));
            const __m128i st = _mm_load_si128((const __m128i*)&m_state[0]);

            __m128i d = _mm_and_si128(m0, st);
            d = _mm_sub_epi32(d, _mm_and_si128(m0, _mm_set1_epi32(static_cast<int>(inp))));
            d = _mm_add_epi32(d, _mm_and_si128(m0, p_vec));
            // d[0] < 2p < 2^17, приводим вычитанием p.
            d = _mm_sub_epi32(d, _mm_and_si128(_mm_cmpgt_epi32(d, _mm_set1_epi32(static_cast<int>(p) - 1)), p_vec));

            // v = d * inv_K[0] mod p (сомножители < p, произведение < p^2 < 2^32).
            __m128i v = _mm_mullo_epi32(d, _mm_load_si128((const __m128i*)&m_inv_K[0]));
            {
                alignas(16) u32x4 tmp;
                _mm_store_si128((__m128i*)&tmp[0], v);
                for (int i=0; i<4; ++i) {
                    tmp[i] %= static_cast<SAMPLE>(p);
                }
                v = _mm_load_si128((const __m128i*)&tmp[0]);
            }
            v = _mm_shuffle_epi32(v, 0x00);

            const __m128i next_st = _mm_srli_si128(st, 4);
            const __m128i next_k = _mm_srli_si128(_mm_load_si128((const __m128i*)&m_K[0]), 4);
            __m128i prod = _mm_mullo_epi32(v, next_k);
            {
                alignas(16) u32x4 tmp;
                _mm_store_si128((__m128i*)&tmp[0], prod);
                for (int i=0; i<4; ++i) {
                    tmp[i] %= static_cast<SAMPLE>(p);
                }
                prod = _mm_load_si128((const __m128i*)&tmp[0]);
            }
            __m128i res = _mm_sub_epi32(next_st, prod);
            res = _mm_add_epi32(res, _mm_and_si128(_mm_cmpgt_epi32(_mm_setzero_si128(), res), p_vec));
            res = _mm_blend_epi16(res, v, 3 << (2 * (m - 1)));
            _mm_store_si128((__m128i*)&m_state[0], res);
        }
#else
        const u32 diff = (m_state[0] >= inp)
            ? static_cast<u32>(m_state[0] - inp)
            : static_cast<u32>(m_state[0]) - static_cast<u32>(inp) + static_cast<u32>(p);
        const u32 m_v = (static_cast<u32>(m_inv_K[0]) * (diff % static_cast<u32>(p))) % static_cast<u32>(p);
        for (int i=0; i<m-1; i++) {
            const u32 subtrahend = (m_v * static_cast<u32>(m_K[i+1])) % static_cast<u32>(p);
            int32_t res = static_cast<int32_t>(m_state[i+1]) - static_cast<int32_t>(subtrahend);
            res += res < 0 ? static_cast<int32_t>(p) : 0;
            m_state[i] = static_cast<SAMPLE>(res);
        }
        m_state[m-1] = static_cast<SAMPLE>(m_v);
#endif
    }

    /**
     * @brief Возвести в квадрат, то есть вычислить состояние (x^s)^2, где
     * x^s - текущее состояние (некоторая степень s вспомогательной переменной x).
     * Соответствует s итерациям next() - прямое вычисление (долго, если s - большое число).
     */
    void square() {
        const STATE old_state = m_state;
        for (int i=0; i<m; ++i) {
            m_state[i] = 0;
        }
        for (int power=2*m-2; power>=0; --power) {
            SAMPLE v = 0;
            for (int i=0; i<power/2 + 1; ++i) {
                const int j = power - i;
                if ((j >= m) || (j < 0)) { continue;}
                const SAMPLE tmp = (old_state[i] * old_state[j]) % SAMPLE(p);
                v += (i != j) ? (2*tmp) % SAMPLE(p) : tmp;
            }
            next(v);
        }
    }

    /**
     * @brief Умножить текущее состояние x^s на некоторое другое состояние x^t.
     * Итоговое состояние генератора становится равным x^(s+t).
     * @param other Другое состояние x^t.
     */
    void mult_by(STATE other) {
        const STATE old_state = m_state;
        for (int i=0; i<m; ++i) {
            m_state[i] = 0;
        }
        for (int power=2*m-2; power>=0; --power) {
            SAMPLE v = 0;
            for (int i=0; i<power+1; ++i) {
                const int j = power - i;
                if ((j >= m) || (j < 0)) { continue;}
                if ((i >= m) || (i < 0)) { continue;}
                v += (old_state[i] * other[j]) % SAMPLE(p);
            }
            next(v);
        }
    }

    /**
     * @brief Насытить генератор.
     * @param q Количество тактов.
     */
    void saturate(int q=m) {
        for (int i=0; i<q; ++i) {
            next();
        }
    }

    /**
     * @brief Является ли заданное состояние текущим состоянием генератора.
     * @param st Заданное состояние.
     * @return Да/нет.
     */
    bool is_state(STATE st) const {
#ifdef USE_SSE
        bool res = true;
        for (int i=0; i<m; ++i) {
            res &= (m_state[i] == st[i]);
        }
        return res;
#else
        return (st == m_state);
#endif
    }

    auto get_state() const {
        return m_state;
    }

    auto get_cell(int idx) const {
        return m_state[idx];
    }
private:
    alignas(16) STATE m_state {};

    alignas(16) STATE m_K {};

    alignas(16) STATE m_inv_K {};

    /**
     * @brief Вычисляется обратный (по умножению) коэффициент.
     */
    void m_calculate_inverse_of_K() {
        const auto x = m_K[0];
        assert(x != 0);
        m_inv_K[0] = 1;
        for (;;) {
            if (((x*m_inv_K[0]) % static_cast<SAMPLE>(p)) == SAMPLE(1)) {
                break;
            }
            m_inv_K[0]++;
        }
    }
};

/**
 * @brief Класс сдвоенного LFSR генератора общей длиной m = 4*2.
 * Хранит числа в 16-битных ячейках.
 * Цель данного генератора: оптимизировать использование основного класса (см. LFSR), если
 * требуется парная работа генераторов. Генераторы работают независимо, но в
 * одном 128-битном регистре.
 */
template <int p>
class LFSR_paired_2x4 {
    static_assert(p < 256);
    static_assert(p > 1);
public:
    /**
     * @brief Конструктор.
     * @param K Коэффициенты двух порождающих полиномов в поле GF(p^4).
     */
    constexpr LFSR_paired_2x4(u16x8 K): m_K(K) {m_calculate_inverse_of_K();};

    void set_state(u16x8 state) {
        m_state = state;
    }

    void set_unit_state() {
        m_state = {1, 0, 0, 0, 1, 0, 0, 0};
    }

    void set_K(u16x8 K) {
        m_K = K;
        m_calculate_inverse_of_K();
    }

    /**
     * @brief Сделать шаг вперед (один такт генератора).
     * @param input Входной символ (по модулю p), который одинаково
     * подается на оба генератора.
     */
    void next(u16 input=0) {
#ifdef USE_SSE
        __m128i a = _mm_set1_epi16(m_state[3]);
        __m128i b = _mm_set1_epi16(m_state[3] ^ m_state[7]);
        b = _mm_slli_si128(b, 8);
        a = _mm_xor_si128(a, b);
        b = _mm_load_si128((const __m128i*)&m_K[0]);
        __m128i c = _mm_mullo_epi16(a, b);

        const __m128i mask = _mm_set_epi16(-1, -1, -1, 0, -1, -1, -1, 0);
        __m128i inp = _mm_andnot_si128( mask, _mm_set1_epi16(input) );

        __m128i d = _mm_load_si128((const __m128i*)&m_state[0]);
        d = _mm_and_si128(mask, _mm_slli_si128(d, 2));
        d = _mm_add_epi16(c, d);
        d = _mm_add_epi16(inp, d);
        _mm_store_si128((__m128i*)&m_state[0], d);
        for (int i=0; i<8; ++i) {
            m_state[i] %= static_cast<u16>(p);
        }
#else
        u16 m_v3 = m_state[3];
        u16 m_v7 = m_state[7];
        for (int i=7; i>4; i--) {
            m_state[i] = (m_state[i-1] + m_v7*m_K[i]) % static_cast<u16>(p);
            m_state[i-4] = (m_state[i-1-4] + m_v3*m_K[i-4]) % static_cast<u16>(p);
        }
        m_state[0] = (input + m_v3*m_K[0]) % static_cast<u16>(p);
        m_state[4] = (input + m_v7*m_K[4]) % static_cast<u16>(p);
#endif
    }

    /**
     * @brief Сделать шаг вперед (один такт генератора).
     * @param inp1 Входной символ (по модулю p) первого генератора.
     * @param inp2 Входной символ (по модулю p) второго генератора.
     */
    void next(u16 inp1, u16 inp2) {
#ifdef USE_SSE
        __m128i a = _mm_set1_epi16(m_state[3]);
        __m128i b = _mm_set1_epi16(m_state[3] ^ m_state[7]);
        b = _mm_slli_si128(b, 8);
        a = _mm_xor_si128(a, b);
        b = _mm_load_si128((const __m128i*)&m_K[0]);
        __m128i c = _mm_mullo_epi16(a, b);

        __m128i mask1 = _mm_slli_si128(_mm_set1_epi16(-1), 2);
        const __m128i mask2 = _mm_set_epi16(-1, -1, -1, 0, -1, -1, -1, -1);
        __m128i inp = _mm_andnot_si128( mask1, _mm_set1_epi16(inp1) );
        inp = _mm_or_si128( inp, _mm_andnot_si128( mask2, _mm_set1_epi16(inp2) ) );

        const __m128i mask = _mm_and_si128(mask1, mask2); // _mm_set_epi16(-1, -1, -1, 0, -1, -1, -1, 0);
        __m128i d = _mm_load_si128((const __m128i*)&m_state[0]);
        d = _mm_and_si128(mask, _mm_slli_si128(d, 2));
        d = _mm_add_epi16(c, d);
        d = _mm_add_epi16(inp, d);
        _mm_store_si128((__m128i*)&m_state[0], d);
        for (int i=0; i<8; ++i) {
            m_state[i] %= static_cast<u16>(p);
        }
#else
        u16 m_v3 = m_state[3];
        u16 m_v7 = m_state[7];
        for (int i=7; i>4; i--) {
            m_state[i] = (m_state[i-1] + m_v7*m_K[i]) % static_cast<u16>(p);
            m_state[i-4] = (m_state[i-1-4] + m_v3*m_K[i-4]) % static_cast<u16>(p);
        }
        m_state[0] = (inp1 + m_v3*m_K[0]) % static_cast<u16>(p);
        m_state[4] = (inp2 + m_v7*m_K[4]) % static_cast<u16>(p);
#endif
    }

    /**
     * @brief Сделать шаг назад (один такт генератора). Обратно к next(inp1, inp2).
     * @param inp1 Входной символ (по модулю p) первого генератора.
     * @param inp2 Входной символ (по модулю p) второго генератора.
     */
    /**
     * @brief Сделать шаг назад (один такт генератора). Обратно к next(inp1, inp2).
     * @param inp1 Входной символ (по модулю p) первого генератора.
     * @param inp2 Входной символ (по модулю p) второго генератора.
     * @details Реализация в стиле AllPass: разности (state[0]-inp1) и (state[4]-inp2)
     * приводятся по модулю p ДО умножения на обратные коэффициенты, поэтому все
     * произведения меньше p^2 < 2^16 и _mm_mullo_epi16 их не усекает.
     * Раньше умножалось (state - inp + p) < 2p, произведение достигало 2*p^2,
     * и при p > 181 коэффициент m_v получался неверным.
     */
    void back(u16 inp1, u16 inp2) {
#ifdef USE_SSE
        const __m128i p_vec = _mm_set1_epi16(static_cast<short>(p));
        const __m128i st = _mm_load_si128((const __m128i*)&m_state[0]);

        // d = {state[0] - inp1 + p, 0,0,0, state[4] - inp2 + p, 0,0,0}.
        // Т.к. state[i] < p и inp < p, вычитание по модулю 2^16 само достраивает +p.
        __m128i d = _mm_sub_epi16(_mm_and_si128(st, _mm_setr_epi16(-1,0,0,0,-1,0,0,0)),
                                  _mm_setr_epi16(static_cast<short>(inp1),0,0,0,static_cast<short>(inp2),0,0,0));
        d = simd_mod_p16<p>(_mm_add_epi16(d, _mm_setr_epi16(static_cast<short>(p),0,0,0,static_cast<short>(p),0,0,0)));

        // v1 = d[0]*inv_K[0] mod p, v2 = d[4]*inv_K[4] mod p.
        __m128i v = simd_mod_p16<p>(_mm_mullo_epi16(d, _mm_load_si128((const __m128i*)&m_inv_K[0])));
        const u16 v1 = static_cast<u16>(_mm_extract_epi16(v, 0));
        const u16 v2 = static_cast<u16>(_mm_extract_epi16(v, 4));
        v = _mm_shuffle_epi8(v, _mm_setr_epi8(0,1, 0,1, 0,1, 0,1, 8,9, 8,9, 8,9, 8,9));

        // state[i] = (state[i+1] - v*K[i+1]) mod p; state[3] = v1, state[7] = v2.
        const __m128i next_st = _mm_srli_si128(st, 2);
        const __m128i next_k = _mm_srli_si128(_mm_load_si128((const __m128i*)&m_K[0]), 2);
        const __m128i prod = simd_mod_p16<p>(_mm_mullo_epi16(v, next_k));
        __m128i res = _mm_sub_epi16(next_st, prod);
        res = _mm_add_epi16(res, _mm_and_si128(_mm_cmpgt_epi16(_mm_setzero_si128(), res), p_vec));
        res = _mm_blend_epi16(res, _mm_setr_epi16(0,0,0,static_cast<short>(v1),0,0,0,static_cast<short>(v2)), 0x88);
        _mm_store_si128((__m128i*)&m_state[0], res);
#else
        const u32 up = static_cast<u32>(p);
        const u32 s0 = static_cast<u32>(m_state[0]);
        const u32 s4 = static_cast<u32>(m_state[4]);
        const u32 i1 = static_cast<u32>(inp1) % up;
        const u32 i2 = static_cast<u32>(inp2) % up;
        const u32 diff1 = (s0 >= i1) ? (s0 - i1) : (s0 + up - i1);
        const u32 diff2 = (s4 >= i2) ? (s4 - i2) : (s4 + up - i2);
        const u32 m_v_1 = (static_cast<u32>(m_inv_K[0]) * diff1) % up;
        const u32 m_v_2 = (static_cast<u32>(m_inv_K[4]) * diff2) % up;
        for (int i=0; i<3; i++) {
            const u32 a = (m_v_1 * static_cast<u32>(m_K[i+1])) % up;
            const u32 b = (m_v_2 * static_cast<u32>(m_K[i+5])) % up;
            m_state[i]   = static_cast<u16>((static_cast<u32>(m_state[i+1]) + up - a) % up);
            m_state[i+4] = static_cast<u16>((static_cast<u32>(m_state[i+5]) + up - b) % up);
        }
        m_state[3] = static_cast<u16>(m_v_1);
        m_state[7] = static_cast<u16>(m_v_2);
#endif
    }

    auto get_state() const {
        return m_state;
    }

    /**
     * @brief Совпадает ли заданное состояние с текущим "нижним" состоянием генератора.
     * @param st Заданное состояние.
     * @return Да/нет.
     */
    bool is_state_low(u16x8 st) const {
        bool res = true;
        for (int i=0; i<4; ++i) {
            res &= (m_state[i] == st[i]);
        }
        return res;
    }

    /**
     * @brief Совпадает ли заданное состояние с текущим "верхним" состоянием генератора.
     * @param st Заданное состояние.
     * @return Да/нет.
     */
    bool is_state_high(u16x8 st) const {
        bool res = true;
        for (int i=0; i<4; ++i) {
            res &= (m_state[i+4] == st[i+4]);
        }
        return res;
    }
private:
    alignas(16) u16x8 m_state {};

    alignas(16) u16x8 m_K {};

    alignas(16) u16x8 m_inv_K {};

    /**
     * @brief Вычисляются обратные (по умножению) коэффициенты.
     */
    void m_calculate_inverse_of_K() {
        const auto x0 = m_K[0];
        const auto x4 = m_K[4];
        assert(x0 != 0);
        assert(x4 != 0);
        m_inv_K[0] = 1;
        m_inv_K[4] = 1;
        bool achieved0 = false;
        bool achieved4 = false;
        for (;;) {
            achieved0 = achieved0 ? achieved0 : ((x0*m_inv_K[0]) % static_cast<u16>(p)) == u16(1);
            achieved4 = achieved4 ? achieved4 : ((x4*m_inv_K[4]) % static_cast<u16>(p)) == u16(1);
            if (achieved0 && achieved4) {
                break;
            }
            m_inv_K[0] += !achieved0;
            m_inv_K[4] += !achieved4;
        }
    }
};

}
