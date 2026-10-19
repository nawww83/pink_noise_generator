#pragma once

namespace qgauss {

template <typename T, typename Generator>
inline T GetQuasiGaussSample(Generator& generator) { // Sum of 4 uniform samples.
    auto samples = generator.next();
    T sample_f = 0;
    for (const auto sample : samples) {
        sample_f += T(sample) / T(65535);
    }
    // Поправка DC подобрана эмпирически: маргинальное распределение 16-битных
    // слов rng2 немного смещено вниз (XOR-микширование в rng2::gens::next()).
    // Измерения на 2*10^8 сэмплов: среднее суммы четырёх слов дает -0.00126
    // (около -1/805); с поправкой 1/833 остаток DC ~ -6e-5, т.е. в ~20 раз меньше.
    // Depends on: tune it for your PRNG.
    constexpr T bias_correction = 1./833.;
    sample_f = T(0.5) * (sample_f - 2) + bias_correction;
    // Диапазон с учетом поправки: (-0.9988, +1.0012), а не (-1, 1).
    return sample_f;
}

}
