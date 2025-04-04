#include "widget.h"
#include "./ui_widget.h"
#include "plotter.h"
#include "noisegenerator.h"
#include "utils.h"
#include <QVector>
#include <QTimer>
#include <random>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrentRun>
#include "rfft.h"
#include "cpuid.h"

using my_float = float;

constexpr int N_samples = 8192;
Q_GLOBAL_STATIC(Plotter, plotter);
Q_GLOBAL_STATIC(NoiseGenerator<my_float>, noise_generator, (8));
Q_GLOBAL_STATIC(QTimer, timer);

namespace {
namespace fft {
    Rfft* g_fft = nullptr;
    Q_GLOBAL_STATIC(Plotter, plotter);
    int len_fft = 0;
    data_t* _in = nullptr;
    data_t* _out = nullptr;
    Q_GLOBAL_STATIC(QVector<my_float>, spm);
}
}

namespace {
namespace rng {
    std::random_device rd{};
    std::mt19937 rnd_generator{rd()};
    std::normal_distribution<my_float> gauss_distribution{0.0, 1.0};
}
}

static QFutureWatcher<void> g_watcher;

static QString g_label_optimize_button;

std::atomic<bool> g_stop_optimization;
std::atomic<bool> g_optimization_failed;

namespace {
namespace opt {
    IirSettings<my_float> g_iir_settings;
    Q_GLOBAL_STATIC(QVector<QPointF>, g_data_1);
    Q_GLOBAL_STATIC(QVector<QPointF>, g_data_2);
    my_float maxY = -std::numeric_limits<my_float>::max();
    my_float minY = std::numeric_limits<my_float>::max();
}
}

static int g_update_interval_ms = 40;

static int prepare_fft(int n) {
    const auto Nfft_est = n;
    auto lN_est = std::log2(Nfft_est);
    auto is_power_two = (static_cast<int>( std::pow(2., std::round(lN_est)) ) == Nfft_est);
    auto lN = (is_power_two) ? static_cast<int>(lN_est + 0.5) : static_cast<int>(lN_est) + 1;

    if (fft::g_fft) {
        auto lN_prev = fft::g_fft->getLogSize();
        if (lN_prev == lN)
            return fft::g_fft->getSize();
    }
    if (fft::g_fft) {
        delete fft::g_fft;
        fft::g_fft = nullptr;
    }

    CPUID cpuID(1);
    const bool isSSE3supported = cpuID.ECX() & 0x1;
    if (isSSE3supported) {
        fft::g_fft = new RfftSSE3;
    } else {
        fft::g_fft = new Rfftx86;
    }

    fft::g_fft->setLogSize(lN);
    return fft::g_fft->getSize();
}

static void allocate_fft_arrays(int n) {
    if (fft::_in) {
        _mm_free(fft::_in);
        fft::_in = nullptr;
    }
    if (fft::_out) {
        _mm_free(fft::_out);
        fft::_out = nullptr;
    }
    fft::_in  = static_cast<data_t*>(_mm_malloc(n * sizeof(data_t), 16));
    fft::_out = static_cast<data_t*>(_mm_malloc(n * sizeof(data_t), 16));

    fft::spm->fill(n/2, 0);
}

static void free_fft_arrays() {
    if (fft::_in) {
        _mm_free(fft::_in);
        fft::_in = nullptr;
    }
    if (fft::_out) {
        _mm_free(fft::_out);
        fft::_out = nullptr;
    }
}

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setFixedSize(this->size().width(), this->size().height());
    ui->spbx_update_interval->setValue(g_update_interval_ms);
    IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
                                        .mTau = {3.933, 1.005, 1.},
                                        .mPowers = {3./2., 5./2., 7./2.},
                                        .mCoeffs = {1., -9./8., 145./128.}};
    noise_generator->SetIirSettings(iir_settings);
    qDebug() << "Size of my float: " << sizeof(my_float);
    if (std::is_same_v<my_float, float>) {
        fft::len_fft = prepare_fft(N_samples);
        allocate_fft_arrays(fft::len_fft);
        qDebug() << "FFT prepared: len: " << fft::len_fft;
    }

    connect(&g_watcher, &QFutureWatcher<void>::finished, this, &Widget::optimizationFinished);
    connect(timer, &QTimer::timeout, this, &Widget::updatePlot);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::closeEvent(QCloseEvent* event) {
    timer->stop();
    free_fft_arrays();
    if (std::is_same_v<my_float, float>) {
        fft::plotter->close();
    }
    plotter->close();
}

void Widget::on_btn_plot_clicked()
{
    if (std::is_same_v<my_float, float>) {
        PlotSettings settings;
        settings.minX = 0;
        settings.maxX = fft::len_fft/2;
        settings.numXTicks = 8;
        settings.minY = -10;
        settings.maxY = 50;
        settings.numYTicks = 6;
        fft::plotter->setWindowTitle("Power Spectral Density");
        fft::plotter->setPlotSettings(settings);
        fft::plotter->clearCurves();
    }
    ui->btn_optimize->setEnabled(false);
    qDebug() << "plot";
    qDebug() << "DC offset: " << (noise_generator->GetDCoffsetCorrectionStatus() ? "On" : "Off");
    PlotSettings settings;
    settings.minX = 0;
    settings.maxX = N_samples;
    settings.numXTicks = 8;
    settings.minY = -20;
    settings.maxY = 20;
    settings.numYTicks = 4;
    plotter->setWindowTitle("Pink Noise");
    plotter->setPlotSettings(settings);
    plotter->clearCurves();
    timer->start(g_update_interval_ms);
}

void Widget::updatePlot()
{
    static QVector<QPointF> data;
    data.clear();
    my_float t = 0;
    for (int i = 0; i < N_samples; ++i) {
        const my_float input = rng::gauss_distribution(rng::rnd_generator);
        const my_float sample = noise_generator->NextSample(input);
        data.emplace_back(t++, sample);
    }
    plotter->setCurveData(0, data);
    plotter->show();
    if (std::is_same_v<my_float, float>) {
        static QVector<QPointF> spm_dB;
        spm_dB.clear();
        for(int k = 0; k < N_samples; ++k) {
            fft::_in[k].real(data.at(k).y());
            fft::_in[k].imag(0);
        }
        for(int k = 0; k < (fft::len_fft - N_samples); ++k) {
            fft::_in[k].real(0);
            fft::_in[k].imag(0);
        }
        fft::g_fft->c2cfft(fft::_in, fft::_out);
        my_float f = 0;
        fft::spm->resize(fft::len_fft/2);
        my_float* spm = fft::spm->data();
        for(int k = 0; k < fft::len_fft/2; ++k) {
            // Усреднение квадратов амплитуд: экспоненциальным фильтром для простоты.
            const my_float Am2 = 2*(fft::_out[k].real() * fft::_out[k].real() + fft::_out[k].imag() * fft::_out[k].imag())/fft::len_fft;
            *spm = *spm * my_float(0.995) + my_float(1. - 0.995) * Am2;
            spm_dB.emplace_back(f++, 10 * std::log10(*spm));
            spm++;
        }
        fft::plotter->setCurveData(0, spm_dB);
        fft::plotter->show();
    }
}

void Widget::on_btn_stop_clicked()
{
    qDebug() << "stopping...";
    timer->stop();
    if (g_watcher.isRunning()) {
        g_stop_optimization.store(true);
        g_watcher.waitForFinished();
    }
    ui->btn_optimize->setEnabled(true);
    qDebug() << "stopped";
}

static void optimize() {
    constexpr bool do_optimization = true;
    constexpr int sampling_factor = 128;
    constexpr int N = N_samples*sampling_factor;
    const my_float initial_parameters[NUM_OF_IIRS] {3.933, 1.005, 1.};
    IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
                                .mTau = {},
                                .mPowers = {3./2., 5./2., 7./2.},
                                .mCoeffs = {1., -9./8., 145./128.}};

    for (int k = 0; k < NUM_OF_IIRS; ++k)
        iir_settings.mTau[k] = initial_parameters[k];
    my_float sum_error = 0.;
    my_float sum_error_c = 0.;
    const my_float dp = 0.001; // parameter step.
    const auto& seq_1_2 = noise_generator->CalculateSequence_1_2(N);
    auto calculate_error = [&seq_1_2](int n) -> my_float {
        my_float result = 0;
        for (int j = 0; j < n; ++j) {
            const my_float input = j == 0;
            const my_float exact_value = seq_1_2.at(j) ;
            const my_float sample = noise_generator->NextSample(input);
            result += std::abs(exact_value - sample);
        }
        return result / my_float(n);
    };
    if (do_optimization) {
        const int num_of_iters = 5;
        for (int iterations = 0; iterations < num_of_iters ; iterations++) {
            qDebug() << "Iterations: " << iterations << " from " << (num_of_iters-1);
            for (int s = 0; s < NUM_OF_IIRS; ++s) {
                qDebug() << "IIR filter index: " << s << " from " << (NUM_OF_IIRS-1);
                noise_generator->SetIirSettings(iir_settings);
                sum_error = calculate_error(N);
                auto& parameter_reference = iir_settings.mTau[s];
                my_float direction = 1;
                for (int repeat = 0; repeat < 4; ) {
                    if (g_stop_optimization.load()) {
                        g_optimization_failed.store(true);
                        return;
                    }
                    parameter_reference += direction*dp;
                    noise_generator->SetIirSettings(iir_settings);
                    sum_error_c = calculate_error(N);
                    if (sum_error_c >= sum_error) {
                        parameter_reference -= direction*dp;
                        direction = -direction;
                        repeat++;
                    } else {
                        sum_error = sum_error_c;
                    }
                }
            }
            const auto relative_error = std::abs(sum_error - sum_error_c)/std::max(std::abs(sum_error), std::abs(sum_error_c));
            qDebug() << "Total relative error: " << relative_error;
            if (relative_error < 0.001 ) {
                g_optimization_failed.store(false);
                break;
            }
        }
        QMutex m;
        m.lock();
        opt::g_iir_settings = iir_settings;
        m.unlock();
    }
    {
        const auto& seq_1_2 = noise_generator->CalculateSequence_1_2(N, sampling_factor);
        QVector<my_float> error_vector(int(N/sampling_factor));
        QMutex m;
        m.lock();
        noise_generator->SetIirSettings(opt::g_iir_settings);
        opt::maxY = -std::numeric_limits<my_float>::max();
        opt::minY = std::numeric_limits<my_float>::max();
        opt::g_data_1->clear();
        opt::g_data_2->clear();
        my_float t = 0;
        for (int i = 0; i < N/sampling_factor; ++i) {
            const my_float input = i*sampling_factor == 0;
            const my_float sample = noise_generator->NextSample(input);
            for (int k = 1; k < sampling_factor; ++k) {
                const my_float input = (i*sampling_factor + k) == 0;
                noise_generator->NextSample(input);
            }
            error_vector[i] = std::abs(seq_1_2.at(i) - sample);
            const my_float sample_Y = 20 * std::log10(std::abs(sample));
            const my_float exact_Y = 20 * std::log10(seq_1_2.at(i));
            opt::maxY = std::max(sample_Y, opt::maxY);
            opt::maxY = std::max(exact_Y, opt::maxY);
            opt::minY = std::min(sample_Y, opt::minY);
            opt::minY = std::min(exact_Y, opt::minY);
            opt::g_data_1->push_back({t, exact_Y});
            opt::g_data_2->push_back({t, sample_Y});
            t += sampling_factor;
        }
        m.unlock();
    }
}

void Widget::on_btn_optimize_clicked()
{
    timer->stop();
    ui->btn_plot->setEnabled(false);
    ui->btn_optimize->setEnabled(false);
    g_optimization_failed.store(false);
    {
        QMutex m;
        m.lock();
        g_label_optimize_button = ui->btn_optimize->text();
        ui->btn_optimize->setText(QString::fromUtf8("Wait..."));
        opt::g_iir_settings = noise_generator->GetIirSettings();
        m.unlock();
    }
    noise_generator->SetDCoffsetCorrection(false);
    g_stop_optimization.store(false);
    QFuture<void> future = QtConcurrent::run(optimize);
    g_watcher.setFuture(future);
}


void Widget::on_spbx_update_interval_editingFinished()
{
    const auto tmp_interval = ui->spbx_update_interval->value();
    if (timer->isActive() && tmp_interval != g_update_interval_ms) {
        g_update_interval_ms = tmp_interval;
        timer->stop();
        timer->start(g_update_interval_ms);
        qDebug() << "Apply new update interval";
    }
}

void Widget::optimizationFinished()
{
    {
        QMutex m;
        m.lock();
        noise_generator->SetDCoffsetCorrection(true);
        noise_generator->SetIirSettings(opt::g_iir_settings);
        ui->btn_optimize->setEnabled(true);
        ui->btn_plot->setEnabled(true);
        ui->btn_optimize->setText(g_label_optimize_button);
        m.unlock();
    }
    if (g_optimization_failed.load()) {
        return;
    }
    {
        QMutex m;
        m.lock();
        qDebug() << "Optimal parameters: ";
        for (int k = 0; k < NUM_OF_IIRS; ++k)
            qDebug() << opt::g_iir_settings.mTau[k] << ", ";
        m.unlock();
    }
    {
        QMutex m;
        m.lock();
        PlotSettings settings;
        settings.minX = 0;
        settings.maxX = opt::g_data_1->last().rx();
        settings.numXTicks = 8;
        plotter_utils::AdjustY(settings, opt::minY, opt::maxY);
        plotter->setPlotSettings(settings);
        plotter->clearCurves();
        plotter->setCurveData(0, *opt::g_data_1);
        plotter->setCurveData(1, *opt::g_data_2);
        plotter->show();
        m.unlock();
    }
}
