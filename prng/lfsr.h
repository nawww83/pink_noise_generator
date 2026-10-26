#pragma once

/**
 * @author Новиков А.В., nawww83@gmail.com.
 *
 * Генератор LFSR в поле GF(p^m) с числом ячеек m = [1, 8] и простым модулем p = [2, 256*256).
 * Реализованы:
 *  - Генератор общего назначения:
 *     - p = [2, 256) для m = [5, 8],
 *     - p = [2, 256*256) для m = [1, 4].
 * Сдвоенный (счетверенный) генератор с фиксированным m = 4, p = [2, 256) находится ниже в этом же файле.
 */

#include <cstdint>
#include <cassert>
#include <array>
#include <type_traits>
#include <utility>
#include <concepts>
#include <limits>
#include <bit>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(USE_NO_SSE)
#ifndef USE_SSE
#define USE_SSE
#endif
#endif

#ifdef USE_SSE
#include <immintrin.h>
#include <smmintrin.h>
#endif

namespace lfsr8
{

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

template <int m>
class MType
{
public:
    typedef typename std::conditional<(m <= 4), u32, u16>::type SAMPLE;
    typedef typename std::conditional<(m <= 4), u32x4, u16x8>::type STATE;
};

template <int p, int m>
inline void modulo_p(typename lfsr8::MType<m>::STATE &state)
{
    for (int i = 0; i < m; i++)
    {
        state[i] %= (u32)p;
    }
    // Векторный путь обрабатывает все ячейки STATE, в том числе неиспользуемые (>= m).
    // Обнуляем их, чтобы get_state() и сравнение состояний не возвращали мусор.
    for (std::size_t i = (std::size_t)m; i < state.size(); i++)
    {
        state[i] = 0;
    }
}

// Вспомогательная функция для скалярного next()
template <int p, int m>
inline void scalar_next(typename lfsr8::MType<m>::STATE &state, const typename lfsr8::MType<m>::STATE &K, typename lfsr8::MType<m>::SAMPLE input)
{
    using SAMPLE = typename lfsr8::MType<m>::SAMPLE;
    const SAMPLE m_v = state[m - 1];
    for (int i = m - 1; i > 0; i--)
    {
        state[i] = (state[i - 1] + (u32)m_v * K[i]) % (u32)p;
    }
    state[0] = (input + (u32)m_v * K[0]) % (u32)p;
}

// Вспомогательная функция для скалярного back()
template <int p, int m>
inline void scalar_back(typename lfsr8::MType<m>::STATE &state, const typename lfsr8::MType<m>::STATE &K, const typename lfsr8::MType<m>::STATE &K_inv, typename lfsr8::MType<m>::SAMPLE input)
{
    using SAMPLE = typename lfsr8::MType<m>::SAMPLE;
    // (state[0] - input) mod p без знаковых и переполняющихся промежуточных значений.
    const u32 diff = ((u32)state[0] + (u32)p - (u32)input) % (u32)p;
    const u32 m_v = ((u32)K_inv[0] * diff) % (u32)p;
    for (int i = 0; i < m - 1; i++)
    {
        const u32 prod = m_v * (u32)K[i + 1];
        const u32 subtrahend = prod % (u32)p;
        int32_t res = (int32_t)state[i + 1] - (int32_t)subtrahend;
        res += res < 0 ? (int32_t)p : (int32_t)0;
        state[i] = (SAMPLE)res;
    }
    state[m - 1] = (SAMPLE)m_v;
}

template <std::unsigned_integral T>
constexpr T safe_ipow(int base, int exp)
{
    T baseT = base;
    T result = 1;
    for (int i = 0; i < exp; ++i)
    {
        // Проверка: не превысит ли следующее умножение предел типа T
        if (i > 0 && baseT > ((std::numeric_limits<T>::max)() / result))
        {
            return 0; // Возвращаем 0 как маркер переполнения
        }
        result *= baseT;
    }
    return result;
}

/**
     * @brief Генератор LFSR общего назначения в поле GF(p^m).
     * @details
     * p должно быть простым числом в интервале:
     *   [2, 256) для длин регистра (4, 8].
     *   [2, 256*256) для длин регистра [1, 4].
     * m - длина регистра, [1, 8].
     */
template <int p, int m>
class LFSR
{
    static_assert(m > 0 && m <= 8, "Ошибка: длина регистра m должна быть в интервале [1, 8]!");
    static_assert(p > 1, "Ошибка: модуль p должен быть больше 1!");
    static_assert((m > 4) ? (p < 256) : (p < 256 * 256),
                  "Ошибка: модуль p вне допустимого интервала для заданной длины регистра!");

    using STATE = typename MType<m>::STATE;
    using SAMPLE = typename MType<m>::SAMPLE;

public:
    /**
         * @brief Конструктор с параметром.
         * @param K Коэффициенты (-g[0], ..., -g[m-1]) порождающего полинома g(x) степени m в поле GF(p).
         * @details Коэффициент при x^m равен 1 и не используется в явном виде.
         * K[0] должен быть ненулевым по модулю p (иначе обратного по умножению коэффициента не существует).
         */
    constexpr LFSR(STATE K) : m_K(K)
    {
        m_calculate_inverse_of_K();
    };

    /**
         * @brief Максимальный период генератора (достигается для примитивного g(x)).
         * @details Для неприводимого, но не примитивного g(x) период делит T_MAX.
         * Для приводимого g(x) период может не делить T_MAX: например, при
         * g(x) = (x - a) * h(x), где h неприводим и примитивен, период равен p^(m-1) - 1,
         * а T_MAX = p^m - 1 = p*(p^(m-1) - 1) + (p - 1), то есть T_MAX % T = p - 1 != 0.
         * Поэтому в таком случае x^T_MAX != 1, и показатель T_MAX - 1 не является
         * обратным (см. inverse_of).
         */
    static constexpr u64 P_POW_M = safe_ipow<u64>(p, m);

    // Проверяем, что переполнения не случилось (P_POW_M не стал 0)
    static_assert(P_POW_M != 0,
                  "Ошибка: Период превышает возможности 64-битного целого числа!");

    static constexpr u64 T_MAX = P_POW_M - 1;

    /**
         * @brief Установить состояние.
         * @param state Состояние.
         * @details Каждая ячейка приводится по модулю p, поэтому в качестве состояния можно
         * передавать произвольные значения (например, сырой сид генератора). Ячейки с индексом
         * >= m обнуляются, чтобы get_state() не возвращал мусор.
         */
    void set_state(STATE state)
    {
        for (int i = 0; i < m; ++i)
        {
            if ((u32)state[i] >= (u32)p)
            {
                state[i] = (SAMPLE)((u32)state[i] % (u32)p);
            }
        }
        for (std::size_t i = (std::size_t)m; i < state.size(); ++i)
        {
            state[i] = 0;
        }
        m_state = std::move(state);
    }

    /**
         * @brief Установить единичное состояние.
         */
    void set_unit_state()
    {
        m_state = {1};
    }

    /**
         * Установить коэффициенты порождающего полинома g(x).
         */
    void set_K(STATE K)
    {
        m_K = K;
        m_calculate_inverse_of_K();
    }

    /**
         * @brief Сделать шаг вперед (осуществить один такт генератора).
         * @param input Входной символ (должен быть приведен по модулю p!), который подается на вход генератора.
         */
    void next(SAMPLE input = 0)
    {
#ifdef USE_SSE
        if constexpr (m > 4)
        {
            __m128i a = _mm_set1_epi16(m_state[m - 1]);
            __m128i b = _mm_load_si128((const __m128i *)&m_K[0]);
            __m128i tmp = _mm_mullo_epi16(a, b);
            __m128i c = _mm_load_si128((const __m128i *)&m_state[0]);
            c = _mm_slli_si128(c, 2);
            __m128i mask = _mm_slli_si128(_mm_set1_epi16(-1), 2);
            __m128i inp = _mm_andnot_si128(mask, _mm_set1_epi16(input));
            c = _mm_add_epi16(c, tmp);
            c = _mm_add_epi16(c, inp);
            _mm_store_si128((__m128i *)&m_state[0], c);
            modulo_p<p, m>(m_state);
        }
        else
        {
            scalar_next<p, m>(m_state, m_K, input);
        }
#else // Для процессора общего назначения.
        scalar_next<p, m>(m_state, m_K, input);
#endif
    }

    /**
         * @brief Сделать шаг назад. Является операцией, обратной к next(input).
         * @param input Входной символ (должен быть приведен по модулю p!), который подается на вход генератора.
         */
    void back(SAMPLE input = 0)
    {
        scalar_back<p, m>(m_state, m_K, m_inv_K, input);
    }

    /**
         * @brief Возвести в квадрат, то есть вычислить состояние (x^s)^2, где
         * x^s - текущее состояние (некоторая степень s вспомогательной переменной x).
         * Соответствует s итерациям next() - прямое вычисление (долго, если s - большое число).
         */
    void square()
    {
        m_state = m_mult(m_state, m_state, m_K);
    }

    /**
         * @brief Умножить текущее состояние x^s на некоторое другое состояние x^t.
         * Итоговое состояние генератора становится равным x^(s+t).
         * @param other Другое состояние x^t.
         * @details Реализуется через m_mult(): свёртка двух многочленов степени m-1 и
         * приведение результата по g(x), см. описание m_mult(). Алиасинг other == m_state
         * безопасен, так как m_mult() не изменяет аргументы до возврата результата.
         */
    void mult_by(const STATE &other)
    {
        m_state = m_mult(m_state, other, m_K);
    }

    /**
         * @brief Возвести текущее состояние генератора в степень.
         * @param q Показатель степени.
         */
    void power_by(u64 q)
    {
        m_state = m_pow(m_state, m_K, q);
    }

    /**
         * @brief Возвести состояние в степень.
         * @param state Состояние x^t.
         * @param K Коэффициенты порождающего полинома g(x).
         * @param q Показатель степени.
         * @return Состояние x^(qt).
         */
    static constexpr STATE power_by(const STATE &state, const STATE &K, u64 q)
    {
        return m_pow(state, K, q);
    }

    /**
         * @brief Вычислить обратное состояние.
         * @param state Состояние x^t (элемент цикла генератора).
         * @param K Коэффициенты порождающего полинома g(x).
         * @param T Период генератора, то есть порядок x в кольце GF(p)[x]/(g(x)) (x^T = 1).
         * @return Состояние, 1/x^t.
         * @details Так как x^T = 1, для любого состояния цикла s = x^t имеем
         * s^(T-1) = x^(tT-t) = (x^T)^t * x^(-t) = x^(-t) = 1/s. Поэтому показатель T-1
         * корректен при любом (в том числе приводимом) g(x), лишь бы K[0] != 0
         * (иначе x необратим и обратного состояния не существует).
         * Подходит любое кратное порядка x. Для неприводимого g(x) можно передавать
         * T = T_MAX = p^m - 1 (примитивность не требуется). Периоды реальных
         * генераторов см. в random_gen.h; для приводимых g(x) T_MAX не подходит.
         */
    static constexpr STATE inverse_of(const STATE &state, const STATE &K, u64 T)
    {
        assert(T > 0);
        return power_by(state, K, (T > 0) ? (T - 1) : 0);
    }

    /**
         * @brief Насытить генератор.
         * @param q Количество тактов.
         */
    void saturate(int q = m)
    {
        for (int i = 0; i < q; ++i)
        {
            next();
        }
    }

    /**
         * @brief Является ли заданное состояние текущим состоянием генератора.
         * @param state Заданное состояние.
         * @return Да/нет.
         */
    bool is_state(const STATE &state) const
    {
#ifdef USE_SSE
        bool result = true;
        for (int i = 0; i < m; ++i)
        {
            result &= (m_state[i] == state[i]);
        }
        return result;
#else
        return (state == m_state);
#endif
    }

    STATE get_generator_coeffs() const
    {
        return m_K;
    }

    /**
         * @brief Коэффициенты для шага назад.
         * @details Значим только элемент [0] (обратный по умножению K[0]), остальные элементы нулевые.
         */
    STATE get_inverse_of_coeffs() const
    {
        return m_inv_K;
    }

    STATE get_state() const
    {
        return m_state;
    }

    SAMPLE get_cell(int idx) const
    {
        assert((idx >= 0) && (idx < m));
        return m_state[idx];
    }

private:
    /**
         * @brief Умножение двух состояний: результат = lhs * rhs по модулю g(x).
         * @details Свёртка двух многочленов степени m-1 и приведение результата по
         * g(x) = x^m + g[0] + g[1]x + ... + g[m-1]x^(m-1), то есть по правилу
         * x^m = K[0] + K[1]x + ... + K[m-1]x^(m-1), где K[i] = -g[i].
         * Приведение выполняется от старших степеней к младшим (схема Горнера), поэтому
         * достаточно (m-1)*m умножений без вызовов next(). Все промежуточные суммы
         * накапливаются в u64 и берутся по модулю p только для коэффициента-делителя
         * и в конце: максимальное значение ячейки не превышает (2m-1)*(p-1)^2, что
         * для разрешенных p и m заведомо меньше 2^64.
         * @param lhs Первый сомножитель.
         * @param rhs Второй сомножитель.
         * @param K Коэффициенты порождающего полинома g(x).
         * @return Произведение состояний.
         */
    static constexpr STATE m_mult(const STATE &lhs, const STATE &rhs, const STATE &K)
    {
        // Коэффициенты произведения (степени 0 .. 2m-2).
        std::array<u64, 2 * m - 1> coeffs{};
        for (int i = 0; i < m; ++i)
        {
            const u64 a = lhs[i];
            if (a == 0)
                continue;
            for (int j = 0; j < m; ++j)
            {
                coeffs[i + j] += a * (u64)rhs[j];
            }
        }

        // Приведение по модулю g(x): x^power = x^(power-m) * x^m.
        for (int power = 2 * m - 2; power >= m; --power)
        {
            const u64 t = coeffs[power] % (u64)p;
            if (t == 0)
                continue;
            for (int i = 0; i < m; ++i)
            {
                coeffs[power - m + i] += t * (u64)K[i];
            }
        }

        STATE res{};
        for (int i = 0; i < m; ++i)
        {
            res[i] = (SAMPLE)(coeffs[i] % (u64)p);
        }
        return res;
    }

    /**
         * @brief Возведение состояния в степень (схема «слева направо»).
         * @details Не требует временного генератора и не копирует состояние на каждом шаге:
         * не более log2(q) возведений в квадрат и (popcount(q)-1) умножений.
         * @param x Состояние x^t.
         * @param K Коэффициенты порождающего полинома g(x).
         * @param q Показатель степени.
         * @return Состояние x^(qt).
         */
    static constexpr STATE m_pow(const STATE &x, const STATE &K, u64 q)
    {
        // x^0 = 1.
        STATE acc{};
        acc[0] = 1;
        if (q == 0)
            return acc;
        // Старший установленный бит обрабатывается без умножения на 1.
        acc = x;
        for (int bit = (int)std::bit_width(q) - 2; bit >= 0; --bit)
        {
            acc = m_mult(acc, acc, K);
            if (((q >> bit) & 1u) != 0u)
            {
                acc = m_mult(acc, x, K);
            }
        }
        return acc;
    }

    /**
         * @brief Состояние генератора.
         */
    alignas(16) STATE m_state{};

    /**
         * @brief Коэффициенты порождающего полинома g(x).
         */
    alignas(16) STATE m_K{};

    /**
         * @brief Коэффициенты регистра для выполнения шага назад.
         */
    alignas(16) STATE m_inv_K{};

    /**
         * @brief Вычисляется обратный (по умножению) коэффициент K[0].
         * @details Расширенный алгоритм Евклида. Если обратного не существует (K[0] == 0 по модулю p
         * или p не простое), m_inv_K[0] остается нулевым, а back() для такого генератора не работает.
         */
    constexpr void m_calculate_inverse_of_K()
    {
        m_inv_K[0] = 0;
        const int64_t modulo = p;
        const int64_t g0 = (int64_t)((u32)m_K[0] % (u32)p);
        assert(g0 != 0);
        int64_t r0 = modulo;
        int64_t r1 = g0;
        int64_t t0 = 0;
        int64_t t1 = 1;
        while (r1 != 0)
        {
            const int64_t q = r0 / r1;
            const int64_t r2 = r0 - q * r1;
            r0 = r1;
            r1 = r2;
            const int64_t t2 = t0 - q * t1;
            t0 = t1;
            t1 = t2;
        }
        if (r0 != 1)
        {
            return; // Обратного по умножению нет.
        }
        m_inv_K[0] = (SAMPLE)(((t0 % modulo) + modulo) % modulo);
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
