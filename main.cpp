#include "widget.h"

#include <QApplication>
#include <QStyleFactory>

int main(int argc, char *argv[])
{
    // Устанавливаем стиль Fusion ДО инициализации QApplication
    QApplication::setStyle(QStyleFactory::create("Fusion"));

    QApplication a(argc, argv);
    Widget w;
    w.show();
    return a.exec();
}
