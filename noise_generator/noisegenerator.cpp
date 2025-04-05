#include "noisegenerator.h"
#include <cmath>

template <typename T>
NoiseGenerator<T>::NoiseGenerator(int fir_order): mFirOrder(fir_order) {
    Update();
}

template<typename T>
T NoiseGenerator<T>::NextSample(T input)
{
    T iir_output = 0;
    int idx_offset = 0;
    for (int k = 0; k < NUM_OF_IIRS; k++) {
        mIirState[0 + idx_offset] = input * std::exp(static_cast<T>(-1)) + mIirState[0 + idx_offset] * std::exp(static_cast<T>(-1));
        T inner_out = mIirState[0 + idx_offset];
        const auto half = mIirSettings.mR[k]/2;
        T attenuation1 = std::exp(-mIirSettings.mTau[k]);
        const T cfactor1 = std::pow(mIirSettings.mTau[k], mIirSettings.mPowers[k]);
        T factor1 = cfactor1;
        for (int j = 1; j <= half; ++j) {
            const T gain = factor1 * attenuation1;
            mIirState[j + idx_offset] = gain * input + mIirState[j + idx_offset] * attenuation1;
            inner_out += mIirState[j + idx_offset];
            factor1 *= cfactor1;
            attenuation1 = std::pow(attenuation1, mIirSettings.mTau[k]);
        }
        T attenuation2 = std::exp(-T(1)/mIirSettings.mTau[k]);
        const T cfactor2 = std::pow(mIirSettings.mTau[k], -mIirSettings.mPowers[k]);
        T factor2 = cfactor2;
        for (int j = 1; j <= half; ++j) {
            const T gain = factor2 * attenuation2;
            mIirState[j + idx_offset + half] = gain * input + mIirState[j + idx_offset + half] * attenuation2;
            inner_out += mIirState[j + idx_offset + half];
            factor2 *= cfactor2;
            attenuation2 = std::pow(attenuation2, T(1)/mIirSettings.mTau[k]);
        }
        inner_out /= mScales[k];
        iir_output += mIirSettings.mCoeffs[k] * inner_out;
        idx_offset += mIirSettings.mR[k];
    }
    const auto norma = std::sqrt(std::numbers::pi);
    iir_output /= norma;
    // DC offset 1 коррекция.
    if (mMakeDCoffsetCorrection_1) {
        constexpr T epsilon = 1.685e-9; // Остаточный DC offset: -130 дБ vs -64 дБ.
        iir_output += (mCorrector[0] - mCorrector[1])*epsilon;
        mCorrector[0] -= mCorrector[0]*T(1.e-6);
        mCorrector[1] -= mCorrector[1]*T(2.e-6);
    }
    //
    for (int i = mFirOrder - 1; i > 0; --i) {
        mFirState[i] = mFirState[i - 1];
    }
    mFirState[0] = input;
    for (int i = 0; i < mFirOrder; ++i) {
        iir_output += mFirCoeffs[i] * mFirState[i];
    }
    T output = mPrevSample + T(1) * input - T(1)/T(2) * mPrevFilters;
    mPrevSample = output;
    mPrevFilters = iir_output;
    mDCoffset += output;
    mSampleCounter++;
    if (mMakeDCoffsetCorrection_2 && ((mSampleCounter % DC_OFFSET_N) == 0)) {
        mDCoffset /= T(DC_OFFSET_N);
        // qDebug() << "DC offset: " << mDCoffset;
        mPrevSample -= mDCoffset; // Коррекция накопленного паразитного смещения.
        mDCoffset = 0;
        mSampleCounter = 0;
    }
    return output;
}

template<typename T>
void NoiseGenerator<T>::SetDCoffsetCorrection_1(bool on)
{
    mMakeDCoffsetCorrection_1 = on;
}

template<typename T>
bool NoiseGenerator<T>::GetDCoffsetCorrectionStatus_1() const
{
    return mMakeDCoffsetCorrection_1;
}

template<typename T>
void NoiseGenerator<T>::SetDCoffsetCorrection_2(bool on)
{
    mSampleCounter = 0;
    mDCoffset = 0;
    mMakeDCoffsetCorrection_2 = on;
}

template<typename T>
bool NoiseGenerator<T>::GetDCoffsetCorrectionStatus_2() const
{
    return mMakeDCoffsetCorrection_2;
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
        if ((sampling_factor == 0) || (sampling_factor > 0 && i % sampling_factor == 0)) {
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
IirSettings<T> NoiseGenerator<T>::GetIirSettings() const
{
    return mIirSettings;
}

template<typename T>
void NoiseGenerator<T>::Update()
{
    mSampleCounter = 0;
    mCorrector[0] = 1.;
    mCorrector[1] = 1.;
    mPrevSample = 0;
    mPrevFilters = 0;
    mDCoffset = 0;
    const auto& c_3_2_exact = CalculateSequence_3_2(mFirOrder);
    int iir_order = std::accumulate(mIirSettings.mR, mIirSettings.mR + NUM_OF_IIRS, 0);
    mIirState.resize(iir_order); mIirState.fill(1);
    mIir_3_2.clear();
    for (int i = 0; i < mFirOrder; ++i) {
        T out = 0;
        int offset = 0;
        for (int k = 0; k < NUM_OF_IIRS; k++) {
            mIirState[0 + offset] *= std::exp(static_cast<T>(-1));
            T inner_out = mIirState[0 + offset];
            const auto half = mIirSettings.mR[k]/2;
            T attenuation1 = std::exp(-mIirSettings.mTau[k]);
            const T cfactor1 = std::pow(mIirSettings.mTau[k], mIirSettings.mPowers[k]);
            T factor1 = cfactor1;
            for (int j = 1; j <= half; ++j) {
                mIirState[j + offset] *= attenuation1;
                const T gain = factor1;
                inner_out += gain * mIirState[j + offset];
                factor1 *= cfactor1;
                attenuation1 = std::pow(attenuation1, mIirSettings.mTau[k]);
            }
            T attenuation2 = std::exp(-T(1)/mIirSettings.mTau[k]);
            const T cfactor2 = std::pow(mIirSettings.mTau[k], -mIirSettings.mPowers[k]);
            T factor2 = cfactor2;
            for (int j = 1; j <= half; ++j) {
                mIirState[j + offset + half] *= attenuation2;
                const T gain = factor2;
                inner_out += gain * mIirState[j + offset + half];
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
    mIirState.resize(iir_order); mIirState.fill(0);
    mIir_3_2.clear();
    mFirState.resize(mFirOrder); mFirState.fill(0);
}

