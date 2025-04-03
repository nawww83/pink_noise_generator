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

using my_float = double;

Q_GLOBAL_STATIC(Plotter, plotter);
Q_GLOBAL_STATIC(NoiseGenerator<my_float>, noise_gen_f32, (8));
Q_GLOBAL_STATIC(QTimer, timer);
std::random_device rd{};
std::mt19937 rnd_generator{rd()};
std::normal_distribution<my_float> gauss_distribution{0.0, 1.0};

static QFutureWatcher<void> g_watcher;
static IirSettings<my_float> g_iir_settings;
static QString g_label_optimize_button;

std::atomic<bool> g_stop_optimization;
std::atomic<bool> g_optimization_failed;

static QVector<QPointF> g_data_1;
static QVector<QPointF> g_data_2;
static my_float maxY = -std::numeric_limits<my_float>::max();
static my_float minY = std::numeric_limits<my_float>::max();

static int g_update_interval_ms = 40;


Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setFixedSize(this->size().width(), this->size().height());
    ui->spinBox->setValue(g_update_interval_ms);
    IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
                                        .mTau = {3.933, 1.005, 1.},
                                        .mPowers = {3./2., 5./2., 7./2.},
                                        .mCoeffs = {1., -9./8., 145./128.}};
    noise_gen_f32->SetIirSettings(iir_settings);

    connect(&g_watcher, &QFutureWatcher<void>::finished, this, &Widget::optimizationFinished);
    connect(timer, &QTimer::timeout, this, &Widget::updatePlot);
}

Widget::~Widget()
{
    delete ui;
}

void Widget::closeEvent(QCloseEvent* event) {
    timer->stop();
    plotter->close();
}

void Widget::on_pushButton_clicked()
{
    ui->pushButton_3->setEnabled(false);
    qDebug() << "plot";
    PlotSettings settings;
    settings.minX = 0;
    settings.maxX = 8192;
    settings.numXTicks = 8;
    settings.minY = -20;
    settings.maxY = 20;
    settings.numYTicks = 4;
    plotter->setPlotSettings(settings);
    plotter->clearCurves();
    timer->start(g_update_interval_ms);
}

void Widget::updatePlot()
{
    constexpr int N = 8192;
    static QVector<QPointF> data;
    data.resize(N);
    for (int i = 0; i < N; ++i) {
        const my_float input = gauss_distribution(rnd_generator);
        const my_float sample = noise_gen_f32->NextSample(input);
        data[i] = {static_cast<my_float>(i), sample};
    }
    plotter->setCurveData(0, data);
    plotter->show();
}

void Widget::on_pushButton_2_clicked()
{
    qDebug() << "stopping...";
    timer->stop();
    if (g_watcher.isRunning()) {
        g_stop_optimization.store(true);
        g_watcher.waitForFinished();
    }
    ui->pushButton_3->setEnabled(true);
    qDebug() << "stopped";
}

static void optimize() {
    constexpr bool do_optimization = true;
    constexpr int sampling_factor = 32;
    constexpr int N = 16384*sampling_factor;
    const my_float init_par[NUM_OF_IIRS] {3.933, 1.005, 1.};
    IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
                                .mTau = {},
                                .mPowers = {3./2., 5./2., 7./2.},
                                .mCoeffs = {1., -9./8., 145./128.}};

    for (int k = 0; k < NUM_OF_IIRS; ++k)
        iir_settings.mTau[k] = init_par[k];
    my_float error_value = 0.;
    my_float error_c = 0.;
    const my_float dp = 0.001;
    QVector<my_float> error(N);
    auto calculate_error = [&error](int N) -> my_float {
        my_float sum_error = 0;
        const auto& seq_1_2 = noise_gen_f32->CalculateSequence_1_2(N);
        for (int j = 0; j < N; ++j) {
            const my_float input = j == 0;
            const my_float sample = noise_gen_f32->NextSample(input);
            error[j] = std::abs(seq_1_2.at(j) - sample);
            sum_error += error.at(j);
        }
        sum_error /= N;
        return sum_error;
    };
    if (do_optimization) {
        for (;;) {
            for (int s = 0; s < NUM_OF_IIRS; ++s) {
                qDebug() << "iir index: " << s;
                noise_gen_f32->SetIirSettings(iir_settings);
                error_value = calculate_error(N);
                auto& par_ref = iir_settings.mTau[s];
                my_float direction = 1;
                for (int repeat = 0; repeat < 3; ) {
                    if (g_stop_optimization.load()) {
                        g_optimization_failed.store(true);
                        qDebug() << "break";
                        return;
                    }
                    par_ref += direction*dp;
                    noise_gen_f32->SetIirSettings(iir_settings);
                    error_c = calculate_error(N);
                    if (error_c >= error_value) {
                        par_ref -= direction*dp;
                        direction = -direction;
                        repeat++;
                    } else {
                        error_value = error_c;
                    }
                }
            }
            const auto rel = std::abs(error_value - error_c)/std::max(std::abs(error_value), std::abs(error_c));
            qDebug() << "total relative error: " << rel;
            if (rel < 0.001 ) {
                g_optimization_failed.store(false);
                break;
            }
        }
        QMutex m;
        m.lock();
        g_iir_settings = iir_settings;
        m.unlock();
    }
    //
    noise_gen_f32->SetIirSettings(iir_settings);
    const auto& seq_1_2 = noise_gen_f32->CalculateSequence_1_2(N, sampling_factor);
    maxY = -std::numeric_limits<my_float>::max();
    minY = std::numeric_limits<my_float>::max();
    {
        QMutex m;
        m.lock();
        QVector<my_float> error_vector(int(N/sampling_factor));
        g_data_1.clear();
        g_data_2.clear();
        for (int i = 0; i < int(N/sampling_factor); ++i) {
            const my_float input = i*sampling_factor == 0;
            const my_float sample = noise_gen_f32->NextSample(input);
            for (int k = 1; k < sampling_factor; ++k) {
                const my_float input = (i*sampling_factor + k) == 0;
                noise_gen_f32->NextSample(input);
            }
            error_vector[i] = std::abs(seq_1_2.at(i) - sample);
            const my_float sample_Y = 20 * std::log10(std::abs(sample));
            const my_float exact_Y = 20 * std::log10(seq_1_2.at(i));
            maxY = std::max(sample_Y, maxY);
            maxY = std::max(exact_Y, maxY);
            minY = std::min(sample_Y, minY);
            minY = std::min(exact_Y, minY);
            g_data_1.push_back({static_cast<my_float>(i*sampling_factor), exact_Y});
            g_data_2.push_back({static_cast<my_float>(i*sampling_factor), sample_Y});
        }
        m.unlock();
    }
}

void Widget::on_pushButton_3_clicked()
{
    timer->stop();
    ui->pushButton->setEnabled(false);
    ui->pushButton_3->setEnabled(false);
    g_optimization_failed.store(false);
    {
        QMutex m;
        m.lock();
        g_label_optimize_button = ui->pushButton_3->text();
        ui->pushButton_3->setText(QString::fromUtf8("Wait..."));
        g_iir_settings = noise_gen_f32->GetIirSettings();
        m.unlock();
    }
    noise_gen_f32->SetDCoffsetCorrection(false);
    g_stop_optimization.store(false);
    QFuture<void> future = QtConcurrent::run(optimize);
    g_watcher.setFuture(future);
}


void Widget::on_spinBox_editingFinished()
{
    const auto tmp_interval = ui->spinBox->value();
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
        noise_gen_f32->SetDCoffsetCorrection(true);
        noise_gen_f32->SetIirSettings(g_iir_settings);
        ui->pushButton_3->setEnabled(true);
        ui->pushButton->setEnabled(true);
        ui->pushButton_3->setText(g_label_optimize_button);
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
            qDebug() << g_iir_settings.mTau[k] << ", ";
        m.unlock();
    }
    {
        QMutex m;
        m.lock();
        PlotSettings settings;
        settings.minX = 0;
        settings.maxX = g_data_1.last().rx();
        settings.numXTicks = 8;
        plotter_utils::AdjustY(settings, minY, maxY);
        plotter->setPlotSettings(settings);
        plotter->clearCurves();
        plotter->setCurveData(0, g_data_1);
        plotter->setCurveData(1, g_data_2);
        plotter->show();
        m.unlock();
    }
}

