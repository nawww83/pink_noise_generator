#pragma once

namespace qgauss {

template <typename T, typename Generator>
inline T GetQuasiGaussSample(Generator& generator) { // Sum of 4 uniform samples.
    auto samples = generator.next();
    T sample_f = 0;
    for (const auto sample : samples) {
        sample_f += T(sample) / T(65535);
    }
    // Поправка DC: после исправления упаковки rng2 (старшие биты ячеек
    // подаются в бит 3 полубайта, а не в бит 0) среднее суммы четырёх
    // 16-битных слов равно -1.006e-4 (замер на 4*10^8 сэмплов, SE 1.4e-5);
    // точная модель по распределению ячеек даёт -1.019e-4. Поправка 1/9814
    // оставляет остаток DC около +1e-6 (прежняя 1/833 оставляла -6e-5).
    // Depends on: tune it for your PRNG.
    constexpr T bias_correction = 1./9814.;
    sample_f = T(0.5) * (sample_f - 2) + bias_correction;
    // Диапазон с учетом поправки: (-0.9999, +1.0001), а не (-1, 1).
    return sample_f;
}

}
