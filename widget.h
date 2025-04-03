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
    void on_pushButton_clicked();

    void updatePlot();

    void on_pushButton_2_clicked();

    void closeEvent(QCloseEvent* event);

    void on_pushButton_3_clicked();

    void on_spinBox_editingFinished();

    void optimizationFinished();

private:
    Ui::Widget *ui;
};
#endif // WIDGET_H
