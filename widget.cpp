#include "widget.h"
#include "./ui_widget.h"
#include "plotter.h"
#include "noisegenerator.h"
#include <QVector>
#include <QTimer>
#include <random>

using my_float = double;

Q_GLOBAL_STATIC(Plotter, plotter);
Q_GLOBAL_STATIC(NoiseGenerator<my_float>, noise_gen_f32, (8));
Q_GLOBAL_STATIC(QTimer, timer);
std::random_device rd{};
std::mt19937 rnd_generator{rd()};
std::normal_distribution<my_float> gauss_distribution{0.0, 1.0};


Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    this->setFixedSize(this->size().width(), this->size().height());
    IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
                                        .mTau = {3.933, 1.005, 1.},
                                        .mPowers = {3./2., 5./2., 7./2.},
                                        .mCoeffs = {1., -9./8., 145./128.}};
    noise_gen_f32->SetIirSettings(iir_settings);
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
    timer->start(40); // ms
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
    qDebug() << "stop";
    timer->stop();
}

[[maybe_unused]] static void optimize() {
    ;
    // constexpr bool do_optimization = false;
    // constexpr int sampling_factor = 512;
    // constexpr int N = 65536*sampling_factor;
    // const my_float init_par[NUM_OF_IIRS] {3.933, 1.005, 1.};
    // IirSettings<my_float> iir_settings {.mR = {21, 7, 5},
    //                             .mTau = {},
    //                             .mPowers = {3./2., 5./2., 7./2.},
    //                             .mCoeffs = {1., -9./8., 145./128.}};

    // for (int k = 0; k < NUM_OF_IIRS; ++k)
    //     iir_settings.mTau[k] = init_par[k];
    // auto calculate_error = []() -> my_float {
    //     my_float sum_error = 0;
    //     QVector<my_float> error(N);
    //     const auto& seq_1_2 = noise_gen_f32->CalculateSequence_1_2(N);
    //     for (int j = 0; j < N; ++j) {
    //         const my_float input = j == 0;
    //         const my_float sample = noise_gen_f32->NextSample(input);
    //         error[j] = std::abs(seq_1_2.at(j) - sample);
    //         sum_error += error.at(j);
    //     }
    //     sum_error /= N;
    //     return sum_error;
    // };
    // my_float error = 0.;
    // my_float error_c = 0.;
    // const my_float dp = 0.001;
    // if (do_optimization) {
    //     for (;;) {
    //         for (int s = 0; s < NUM_OF_IIRS; ++s) {
    //             noise_gen_f32->SetIirSettings(iir_settings);
    //             error = calculate_error();
    //             auto& par_ref = iir_settings.mTau[s];
    //             my_float dir = 1;
    //             qDebug() << "Error: " << error << ", par: " << par_ref << ", s: " << s;
    //             for (int repeat = 0; repeat < 3; ) {
    //                 par_ref += dir*dp;
    //                 noise_gen_f32->SetIirSettings(iir_settings);
    //                 error_c = calculate_error();
    //                 qDebug() << "par: " << par_ref << ", new error: " << error_c << ", old error: " << error << ", dir: " << dir << ", repeat: " << repeat;
    //                 if (error_c >= error) {
    //                     par_ref -= dir*dp;
    //                     dir = -dir;
    //                     repeat++;
    //                 } else {
    //                     error = error_c;
    //                 }
    //             }
    //         }
    //         const auto rel = std::abs(error - error_c)/std::max(std::abs(error), std::abs(error_c));
    //         qDebug() << "rel: " << rel;
    //         if (rel < 0.001 ) {
    //             break;
    //         }
    //     }
    //     qDebug() << "Optimal parameters: ";
    //     for (int k = 0; k < NUM_OF_IIRS; ++k)
    //         qDebug() << iir_settings.mTau[k] << ", ";
    // }
    // //
    // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    // noise_gen_f32->SetIirSettings(iir_settings);
    // const auto& seq_1_2 = noise_gen_f32->CalculateSequence_1_2(N, sampling_factor);
    // QVector<QPointF> data_2;
    // QVector<QPointF> data_3;
    // my_float maxY = -std::numeric_limits<my_float>::max();
    // my_float minY = std::numeric_limits<my_float>::max();
    // QVector<my_float> error_v(int(N/sampling_factor));
    // for (int i = 0; i < int(N/sampling_factor); ++i) {
    //     const my_float input = i*sampling_factor == 0;
    //     const my_float sample = noise_gen_f32->NextSample(input);
    //     for (int k = 1; k < sampling_factor; ++k) {
    //         const my_float input = (i*sampling_factor + k) == 0;
    //         noise_gen_f32->NextSample(input);
    //     }
    //     error_v[i] = std::abs(seq_1_2.at(i) - sample);
    //     const my_float sample_Y = 20 * std::log10(std::abs(sample));
    //     const my_float exact_Y = 20 * std::log10(seq_1_2.at(i));
    //     maxY = std::max(sample_Y, maxY);
    //     maxY = std::max(exact_Y, maxY);
    //     minY = std::min(sample_Y, minY);
    //     minY = std::min(exact_Y, minY);
    //     data_2.push_back({static_cast<my_float>(i*sampling_factor), exact_Y});
    //     data_3.push_back({static_cast<my_float>(i*sampling_factor), sample_Y});
    // }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // qDebug() << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]";
    // // qDebug() << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]";
    // //
    // // qDebug() << "maxY: " << maxY << ", minY: " << minY;
    // PlotSettings settings;
    // settings.minX = 0;
    // settings.maxX = N;
    // settings.numXTicks = 8;
    // plotter_utils::AdjustY(settings, minY, maxY);
    // plotter->setPlotSettings(settings);
    // plotter->clearCurves();
    // plotter->setCurveData(0, data_2);
    // plotter->setCurveData(0, data_3);
    // plotter->show();
}
