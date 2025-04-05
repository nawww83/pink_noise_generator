#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QCloseEvent>

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();

private slots:
    void on_btn_plot_clicked();

    void updatePlot();

    void on_btn_stop_clicked();

    void closeEvent(QCloseEvent* event);

    void on_btn_optimize_clicked();

    void on_spbx_update_interval_editingFinished();

    void optimizationFinished();

    void plotIrFinished();

    void on_btn_plot_ir_clicked();

private:
    Ui::Widget *ui;
};
#endif // WIDGET_H
