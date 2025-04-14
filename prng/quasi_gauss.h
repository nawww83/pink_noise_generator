#pragma once

namespace qgauss {

template <typename T, typename Generator>
inline T GetQuasiGaussSample(Generator& generator) { // Sum of 4 uniform samples.
    auto samples = generator.next();
    T g_sample_f = 0;
    g_sample_f += T(samples[0]) / T(65535);
    g_sample_f += T(samples[1]) / T(65535);
    g_sample_f += T(samples[2]) / T(65535);
    g_sample_f += T(samples[3]) / T(65535);
    constexpr T bias_correction = 1./833.; // Depends on: tune it for your PRNG.
    g_sample_f = T(0.5) * (g_sample_f - 2) + bias_correction;
    return g_sample_f; // In range (-1, 1).
}

}
