#pragma once

#include <QVector>

/**
 * @brief Количество БИХ-фильтров, генерирующих заданный степенной закон.
 */
constexpr int NUM_OF_IIRS = 3;

/**
 * @brief Количество отсчетов на который применяется коррекция
 * паразитного DC offset, и так далее кратно этому размеру.
 */
constexpr size_t DC_OFFSET_N = 1ull << 21;

/**
 * @brief Параметры БИХ-фильтра, приближенно генерирующего ИХ "3/2".
 * ИХ "3/2" - это ИХ фильтра, интегрирование которой дает ИХ генератора розового шума.
 */
template <typename T>
struct IirSettings {
    int mR[NUM_OF_IIRS]; // Количество экспонент, нечетное число.
    T mTau[NUM_OF_IIRS]; // Параметр-масштаб, как правило в районе 1...5.
    T mPowers[NUM_OF_IIRS]; // Степень p аппроксимируемого закона (1/n)^p.
    T mCoeffs[NUM_OF_IIRS]; // Коэффициенты разложения ИХ "3/2" в ряд по большому индексу n.
};

/**
 * @brief Настройки фильтра-корректора DC offset 1.
 * K(z) = epsilon/(1 - alpha1 * z^-1) - epsilon/(1 - alpha2 * z^-1).
 */
template <typename T>
struct DCoffsetSettings {
    T mAlpha1_complement; // Дополненный до единицы коэффициент геометрической прогрессии 1.
    T mAlpha2_complement; // Дополненный до единицы коэффициент геометрической прогрессии 2.
    T mEpsilon; // Усиление.
    // ac = 1 - a = 2E-6
    // ac = 1 - a = 4E-6
    // ε = 3.36750E-9
};

/**
 * @brief Генератор розового шума.
 */
template <typename T>
class NoiseGenerator
{
public:
    /**
     * @brief Конструктор.
     * @param fir_order Порядок КИХ-фильтра.
     */
    explicit NoiseGenerator(int fir_order);

    /**
     * @brief Получить следующий отсчет розового шума.
     * @param input Отсчет белого (как правило, гауссовского) шума.
     * @return Отсчет розового шума.
     */
    T NextSample(T input);

    /**
     * @brief Включить/выключить коррекцию DC offset 1.
     * Для оптимизации фильтра коррекцию следует выключить, чтобы получать
     * реальные ИХ фильтра.
     * @param on Вкл/выкл.
     */
    void SetDCoffsetCorrection_1(bool on);

    /**
     * @brief Включить/выключить коррекцию DC offset 2.
     * Для оптимизации фильтра коррекцию следует выключить, чтобы получать
     * реальные ИХ фильтра. Для рабочего режима следует включить для
     * избегания долгосрочного паразитного смещения уровня.
     * @param on Вкл/выкл.
     */
    void SetDCoffsetCorrection_2(bool on);

    /**
     * @brief SetDCoffsetSettings_1 Установить параметры фильтра-корректора DC offset 1.
     * @param settings Настройки.
     */
    void SetDCoffsetSettings_1(DCoffsetSettings<T> settings);

    /**
     * @brief Получить статус коррекции DC offset 1.
     * @return Включено/Выключено.
     */
    bool GetDCoffsetCorrectionStatus_1() const;

    /**
     * @brief Получить статус коррекции DC offset 2.
     * @return Включено/Выключено.
     */
    bool GetDCoffsetCorrectionStatus_2() const;

    /**
     * @brief Вспомогательные методы.
     */
    QVector<T> CalculateSequence_3_2(int len);
    QVector<T> CalculateSequence_1_2(int len, int sampling_factor = 0);
    QVector<T> GetIir3_2() const ;
    QVector<T> GetFirCoeffs() const ;

    /**
     * @brief Установить параметры БИХ-фильтра.
     * @param settings Параметры БИХ-фильтра.
     */
    void SetIirSettings(IirSettings<T> settings);

    /**
     * @brief Получить параметры БИХ-фильтра.
     * @return Параметры БИХ-фильтра.
     */
    IirSettings<T> GetIirSettings() const;
private:
    /**
     * @brief Порядок КИХ-фильтра.ъ
     */
    int mFirOrder;

    /**
     * @brief Параметры основного БИХ-фильтра.
     */
    IirSettings<T> mIirSettings {.mR = {15, 5, 5},
                             .mTau = {4.2, 1.2, 1.},
                             .mPowers = {3./2., 5./2., 7./2.},
                             .mCoeffs = {1., -9./8., 145./128.}};
    /**
     * @brief Коэффициенты КИХ-фильтра небольшого порядка, который
     * корректирует ИХ "3/2" в области малых индексов.
     */
    QVector<T> mFirCoeffs;

    /**
     * @brief Регистр-состояние КИХ-фильтра небольшого порядка.
     */
    QVector<T> mFirState;

    /**
     * @brief Масштабирующие коэффициенты. Масштабируют сумму экспонент.
     */
    T mScales[NUM_OF_IIRS];

    /**
     * @brief Регистр основного БИХ-фильтра.
     */
    QVector<T> mIirState;

    /**
     * @brief Приближенная ИХ фильтра "3/2".
     */
    QVector<T> mIir_3_2;

    /**
     * @brief Регистр фильтра-корректора DC offset.
     */
    T mCorrector[2] = {1., 1.};

    /**
     * @brief Настройки фильтра-корректора DC offset 1.
     */
    DCoffsetSettings<T> mDCoffsetSettings_1;

    /**
     * @brief Предыдущий отсчет. Вспомогательная переменная.
     */
    T mPrevSample = 0;

    /**
     * @brief Предыдущий отсчет. Вспомогательная переменная.
     */
    T mPrevFilters = 0;

    /**
     * @brief Текущее смещение уровня шума.
     */
    T mDCoffset = 0;

    /**
     * @brief Счетчик количества отсчетов для контроля DC offset.
     */
    size_t mSampleCounter = 0;

    /**
     * @brief Делать коррекцию DC offset 1.
     */
    bool mMakeDCoffsetCorrection_1 = true;

    /**
     * @brief Делать коррекцию DC offset 2.
     */
    bool mMakeDCoffsetCorrection_2 = true;

    /**
     * @brief Сбросить состояния и пересчитать требуемые ИХ.
     */
    void ResetAll();
};

template class NoiseGenerator<float>;
template class NoiseGenerator<double>;
