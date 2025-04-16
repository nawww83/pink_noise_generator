#pragma once

namespace qgauss {

template <typename T, typename Generator>
inline T GetQuasiGaussSample(Generator& generator) { // Sum of 4 uniform samples.
    auto samples = generator.next();
    T sample_f = 0;
    for (const auto sample : samples) {
        sample_f += T(sample) / T(65535);
    }
    constexpr T bias_correction = 1./833.; // Depends on: tune it for your PRNG.
    sample_f = T(0.5) * (sample_f - 2) + bias_correction;
    return sample_f; // In range (-1, 1).
}

}
