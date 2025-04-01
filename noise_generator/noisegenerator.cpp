#include "noisegenerator.h"
#include <cmath>

template <typename T>
NoiseGenerator<T>::NoiseGenerator(int fir_order): mFirOrder(fir_order) {
    Update();
}

template<typename T>
T NoiseGenerator<T>::NextSample(T input)
{
    T value = 0;
    int offset = 0;
    for (int k = 0; k < NUM_OF_IIRS; k++) {
        mIirOutput[0 + offset] = input * std::exp(static_cast<T>(-1)) + mIirOutput[0 + offset] * std::exp(static_cast<T>(-1));
        T inner_out = mIirOutput[0 + offset];
        const auto half = mIirSettings.mR[k]/2;
        T attenuation1 = std::exp(-mIirSettings.mTau[k]);
        const T cfactor1 = std::pow(mIirSettings.mTau[k], mIirSettings.mPowers[k]);
        T factor1 = cfactor1;
        for (int j = 1; j <= half; ++j) {
            const T gain = factor1 * attenuation1;
            mIirOutput[j + offset] = gain * input + mIirOutput[j + offset] * attenuation1;
            inner_out += mIirOutput[j + offset];
            factor1 *= cfactor1;
            attenuation1 = std::pow(attenuation1, mIirSettings.mTau[k]);
        }
        T attenuation2 = std::exp(-T(1)/mIirSettings.mTau[k]);
        const T cfactor2 = std::pow(mIirSettings.mTau[k], -mIirSettings.mPowers[k]);
        T factor2 = cfactor2;
        for (int j = 1; j <= half; ++j) {
            const T gain = factor2 * attenuation2;
            mIirOutput[j + offset + half] = gain * input + mIirOutput[j + offset + half] * attenuation2;
            inner_out += mIirOutput[j + offset + half];
            factor2 *= cfactor2;
            attenuation2 = std::pow(attenuation2, T(1)/mIirSettings.mTau[k]);
        }
        inner_out /= mScales[k];
        value += mIirSettings.mCoeffs[k] * inner_out;
        offset += mIirSettings.mR[k];
    }
    const auto norma = std::sqrt(std::numbers::pi);
    value /= norma;
    // DC offset коррекция.
    // constexpr T epsilon = 1.71e-9; // new DC offset -120 dB vs -64 dB: no change sign.
    constexpr T epsilon = 1.90e-9; // changed sign: overload.
    value += (mCorrector[0] - mCorrector[1])*epsilon;
    mCorrector[0] -= mCorrector[0]*T(1.e-6);
    mCorrector[1] -= mCorrector[1]*T(2.e-6);
    //
    for (int i = mFirOrder - 1; i > 0; --i) {
        mFirState[i] = mFirState[i - 1];
    }
    mFirState[0] = input;
    for (int i = 0; i < mFirOrder; ++i) {
        value += mFirCoeffs[i] * mFirState[i];
    }
    T output = mPrevSample + T(1) * input - T(1)/T(2) * mPrevFilters;
    mPrevSample = output;
    mPrevFilters = value;
    return output;
}

template<typename T>
QVector<T> NoiseGenerator<T>::CalculateSequence_3_2(int len)
{
    QVector<T> result{1};
    for (int i = 1; i < len; ++i) {
        const T value = result[i-1] * (T(1) - T(3)/T(2)/(i + T(1)));
        result.push_back(value);
    }
    return result;
}

template<typename T>
QVector<T> NoiseGenerator<T>::CalculateSequence_1_2(int len, int sampling_factor)
{
    QVector<T> result{1};
    T prev_value = result.at(0);
    for (int i = 1; i < len; ++i) {
        const T value = prev_value * (T(1) - T(1)/T(2)/(i + T(0)));
        prev_value = value;
        if (sampling_factor > 0 && i % sampling_factor == 0) {
            result.push_back(value);
        }
    }
    return result;
}


template<typename T>
QVector<T> NoiseGenerator<T>::GetIir3_2() const
{
    return mIir_3_2;
}

template<typename T>
QVector<T> NoiseGenerator<T>::GetFirCoeffs() const
{
    return mFirCoeffs;
}

template<typename T>
void NoiseGenerator<T>::SetIirSettings(IirSettings<T> settings)
{
    mIirSettings = settings;
    Update();
}

template<typename T>
void NoiseGenerator<T>::Update()
{
    mCorrector[0] = 1.;
    mCorrector[1] = 1.;
    mPrevSample = 0;
    mPrevFilters = 0;
    const auto& c_3_2_exact = CalculateSequence_3_2(mFirOrder);
    int iir_order = std::accumulate(mIirSettings.mR, mIirSettings.mR + NUM_OF_IIRS, 0);
    mIirOutput.resize(iir_order); mIirOutput.fill(1);
    mIir_3_2.clear();
    for (int i = 0; i < mFirOrder; ++i) {
        T out = 0;
        int offset = 0;
        for (int k = 0; k < NUM_OF_IIRS; k++) {
            mIirOutput[0 + offset] *= std::exp(static_cast<T>(-1));
            T inner_out = mIirOutput[0 + offset];
            const auto half = mIirSettings.mR[k]/2;
            T attenuation1 = std::exp(-mIirSettings.mTau[k]);
            const T cfactor1 = std::pow(mIirSettings.mTau[k], mIirSettings.mPowers[k]);
            T factor1 = cfactor1;
            for (int j = 1; j <= half; ++j) {
                mIirOutput[j + offset] *= attenuation1;
                const T gain = factor1;
                inner_out += gain * mIirOutput[j + offset];
                factor1 *= cfactor1;
                attenuation1 = std::pow(attenuation1, mIirSettings.mTau[k]);
            }
            T attenuation2 = std::exp(-T(1)/mIirSettings.mTau[k]);
            const T cfactor2 = std::pow(mIirSettings.mTau[k], -mIirSettings.mPowers[k]);
            T factor2 = cfactor2;
            for (int j = 1; j <= half; ++j) {
                mIirOutput[j + offset + half] *= attenuation2;
                const T gain = factor2;
                inner_out += gain * mIirOutput[j + offset + half];
                factor2 *= cfactor2;
                attenuation2 = std::pow(attenuation2, T(1)/mIirSettings.mTau[k]);
            }
            if (i == 0) {
                mScales[k] = inner_out;
                inner_out = 1;
            } else {
                inner_out /= mScales[k];
            }
            out += mIirSettings.mCoeffs[k] * inner_out;
            offset += mIirSettings.mR[k];
        }
        mIir_3_2.push_back(out);
    }
    const auto norma = std::sqrt(std::numbers::pi);
    for (auto& el : mIir_3_2) {
        el /= norma;
    }
    mFirCoeffs.clear();
    for (int i = 0; i < mIir_3_2.size(); ++i) {
        mFirCoeffs.push_back(c_3_2_exact.at(i) - mIir_3_2.at(i));
    }
    mIirOutput.resize(iir_order); mIirOutput.fill(0);
    mIir_3_2.clear();
    mFirState.resize(mFirOrder); mFirState.fill(0);
}

