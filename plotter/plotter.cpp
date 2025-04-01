#include <QToolButton>
#include <QtEvents>
#include <QStylePainter>
#include <QStyleOptionFocusRect>
#include <QPainter>
#include <qmath.h>
#include <limits>
#include "plotter.h"


static bool AreTheSame(double a, double b, double scale = 1) {
    return std::fabs(a - b) < 10 * qMax(1., scale) * std::numeric_limits<double>::epsilon();
}


Plotter::Plotter(QWidget *parent) :
    QWidget(parent)
{
    setBackgroundRole(QPalette::Light);
    setAutoFillBackground(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setFocusPolicy(Qt::StrongFocus);
    rubberBandIsShown = false;
    {
        zoomInButton = new QToolButton(this);
        const QPixmap icon_map("://zoomin.png");
        zoomInButton->setIcon(QIcon(icon_map));
        zoomInButton->adjustSize();
        connect(zoomInButton, SIGNAL(clicked()), this, SLOT(zoomIn()));
    }

    {
        zoomOutButton = new QToolButton(this);
        const QPixmap icon_map("://zoomout.png");
        zoomOutButton->setIcon(QIcon(icon_map));
        zoomOutButton->adjustSize();
        connect(zoomOutButton, SIGNAL(clicked()), this, SLOT(zoomOut()));
    }
    setPlotSettings(PlotSettings());
    resize(700, 300);
}

void Plotter::setPlotSettings(const PlotSettings &settings)
{
    // if set settings is equal first settings
    if (!zoomStack.isEmpty()) {
        PlotSettings ps = zoomStack.first();
        if (settings.isEqual(ps))
            return;
    }
    zoomStack.clear();
    zoomStack.append(settings);
    curZoom = 0;
    zoomInButton->hide();
    zoomOutButton->hide();
    refreshPixmap();
}

void Plotter::setCurveData(int id, const QVector<QPointF> &data) {
    curveMap[id] = data;
    refreshPixmap();
}

void Plotter::clearCurve(int id) {
    curveMap.remove(id);
    refreshPixmap();
}

void Plotter::clearCurves() {
    QMutableMapIterator<int, QVector<QPointF> > i(curveMap);
    while (i.hasNext()) {
        i.next()->clear();
    }
    curveMap.clear();
}

QSize Plotter::minimumSizeHint() const {
    return QSize(6 * Margin, 4 * Margin);
}

QSize Plotter::sizeHint() const {
    return QSize(12 * Margin, 4 * Margin);
}

void Plotter::zoomIn() {
    if (curZoom < zoomStack.count() - 1) {
        ++curZoom;
        zoomInButton->setEnabled(curZoom < zoomStack.count() - 1);
        zoomOutButton->setEnabled(true);
        zoomOutButton->show();
        refreshPixmap();
    }
}

void Plotter::zoomOut() {
    if (curZoom > 0) {
        --curZoom;
        zoomOutButton->setEnabled(curZoom > 0);
        zoomInButton->setEnabled(true);
        zoomInButton->show();
        refreshPixmap();
    }
}

void Plotter::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event)

    QStylePainter painter(this);
    painter.drawPixmap(0, 0, pixmap);

    if (rubberBandIsShown) {
        painter.setPen(palette().dark().color());
        painter.drawRect(rubberBandRect.normalized()
                                       .adjusted(0, 0, -1, -1));
    }

    if (hasFocus()) {
        QStyleOptionFocusRect option;
        option.initFrom(this);
        option.backgroundColor = palette().dark().color();
        painter.drawPrimitive(QStyle::PE_FrameFocusRect, option);
    }
}

void Plotter::resizeEvent(QResizeEvent *event)
{
    Q_UNUSED(event)

    int x = width() - (zoomInButton->width()
                       + zoomOutButton->width() + 10);
    zoomInButton->move(x, 5);
    zoomOutButton->move(x + zoomInButton->width() + 5, 5);
    refreshPixmap();
}

void Plotter::mousePressEvent(QMouseEvent *event)
{
    QRect rect(Margin, Margin, width() - 2 * Margin, height() - 2 * Margin);

    if (event->button() == Qt::LeftButton) {
        if (rect.contains(event->pos())) {
            rubberBandIsShown = true;
            rubberBandRect.setTopLeft(event->pos());
            rubberBandRect.setBottomRight(event->pos());
            updateRubberBandRegion();
            setCursor(Qt::CrossCursor);
        }
    }
}

void Plotter::mouseMoveEvent(QMouseEvent *event)
{
    if (rubberBandIsShown) {
        updateRubberBandRegion();
        rubberBandRect.setBottomRight(event->pos());
        updateRubberBandRegion();
    }
}

void Plotter::mouseReleaseEvent(QMouseEvent *event)
{
    if ((event->button() == Qt::LeftButton) && rubberBandIsShown) {
        rubberBandIsShown = false;
        updateRubberBandRegion();
        unsetCursor();

        QRect rect = rubberBandRect.normalized();
        if (rect.width() < 4 || rect.height() < 4)
            return;
        rect.translate(-Margin, -Margin);

        PlotSettings prevSettings = zoomStack[curZoom];
        PlotSettings settings;
        double dx = prevSettings.spanX() / (width() - 2 * Margin);
        double dy = prevSettings.spanY() / (height() - 2 * Margin);
        settings.minX = prevSettings.minX + dx * rect.left();
        settings.maxX = prevSettings.minX + dx * rect.right();
        settings.minY = prevSettings.maxY - dy * rect.bottom();
        settings.maxY = prevSettings.maxY - dy * rect.top();
        settings.adjust();

        zoomStack.resize(curZoom + 1);
        zoomStack.append(settings);
        zoomIn();
    }
}

void Plotter::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Plus:
        zoomIn();
        break;
    case Qt::Key_Minus:
        zoomOut();
        break;
    case Qt::Key_Left:
        zoomStack[curZoom].scroll(-1, 0);
        refreshPixmap();
        break;
    case Qt::Key_Right:
        zoomStack[curZoom].scroll(+1, 0);
        refreshPixmap();
        break;
    case Qt::Key_Down:
        zoomStack[curZoom].scroll(0, -1);
        refreshPixmap();
        break;
    case Qt::Key_Up:
        zoomStack[curZoom].scroll(0, +1);
        refreshPixmap();
        break;
    default:
        QWidget::keyPressEvent(event);
    }
}

void Plotter::wheelEvent(QWheelEvent *event)
{
    int numDegrees = event->angleDelta().y() / 8;
    int numTicks = numDegrees / 15;
    if (event->angleDelta().x() != 0) {
        zoomStack[curZoom].scroll(numTicks, 0);
    } else {
        zoomStack[curZoom].scroll(0, numTicks);
    }
    refreshPixmap();
}

void Plotter::updateRubberBandRegion()
{
    QRect rect = rubberBandRect.normalized();
    update(rect.left(), rect.top(), rect.width(), 1);
    update(rect.left(), rect.top(), 1, rect.height());
    update(rect.left(), rect.bottom(), rect.width(), 1);
    update(rect.right(), rect.top(), 1, rect.height());
}

void Plotter::refreshPixmap()
{
    pixmap = QPixmap(size());
    pixmap.fill(QColor(255, 255, 255));

    QPainter painter(&pixmap);
    drawGrid(&painter);
    drawCurves(&painter);
    update();
}

void Plotter::drawGrid(QPainter *painter)
{
    QRect rect(Margin, Margin,
               width() - 2 * Margin, height() - 2 * Margin);
    if (!rect.isValid())
        return;

    PlotSettings settings = zoomStack[curZoom];
    QPen grid_pen(Qt::green);
    QPen box_pen(Qt::black);

    for (int i = 0; i <= settings.numXTicks; ++i) {
         int x = rect.left() + (i * (rect.width() - 1)
                                  / settings.numXTicks);
         double label = settings.minX + (i * settings.spanX()
                                           / settings.numXTicks);
         if (AreTheSame(label, 0, settings.numXTicks))
             label = 0.;
         painter->setPen(grid_pen);
         painter->drawLine(x, rect.top(), x, rect.bottom());
         painter->setPen(box_pen);
         painter->drawLine(x, rect.bottom(), x, rect.bottom() + 5);
         painter->drawText(x - 50, rect.bottom() + 5, 100, 20,
                              Qt::AlignHCenter | Qt::AlignTop,
                              QString::number(label));
    }
    for (int j = 0; j <= settings.numYTicks; ++j) {
        int y = rect.bottom() - (j * (rect.height() - 1)
                                   / settings.numYTicks);
        double label = settings.minY + (j * settings.spanY()
                                          / settings.numYTicks);
        if (AreTheSame(label, 0, settings.numYTicks))
            label = 0.;
        painter->setPen(grid_pen);
        painter->drawLine(rect.left(), y, rect.right(), y);
        painter->setPen(box_pen);
        painter->drawLine(rect.left() - 5, y, rect.left(), y);
        painter->drawText(rect.left() - Margin, y - 10, Margin - 5, 20,
                             Qt::AlignRight | Qt::AlignVCenter,
                             QString::number(label));
    }
    painter->drawRect(rect.adjusted(0, 0, -1, -1));
}

void Plotter::drawCurves(QPainter *painter)
{
    static const QColor colorForIds[6] = {
        Qt::blue, Qt::red, Qt::magenta, Qt::cyan, Qt::green, Qt::yellow
    };    

    PlotSettings settings = zoomStack[curZoom];
    QRect rect(Margin, Margin,
               width() - 2 * Margin, height() - 2 * Margin);
    if (!rect.isValid())
        return;

    painter->setClipRect(rect.adjusted(+1, +1, -1, -1));

    QMapIterator<int, QVector<QPointF> > i(curveMap);
    while (i.hasNext()) {
        i.next();

        int id = i.key();
        QVector<QPointF> data = i.value();
        QPolygonF polyline(data.count());
        for (int j = 0; j < data.count(); ++j) {
            double dx = data[j].x() - settings.minX;
            double dy = data[j].y() - settings.minY;
            double x = rect.left() + (dx * (rect.width() - 1)
                                         / settings.spanX());
            double y = rect.bottom() - (dy * (rect.height() - 1)
                                           / settings.spanY());
            polyline[j] = QPointF(x, y);
        }
        QPen pen(QBrush(colorForIds[uint(id) % 6], Qt::SolidPattern), 2.0);
        painter->setPen(pen);
        painter->drawPolyline(polyline);
    }
}


PlotSettings::PlotSettings()
{
    minX = 0.0;
    maxX = 10.0;
    numXTicks = 5;

    minY = 0.0;
    maxY = 10.0;
    numYTicks = 4;
}

bool PlotSettings::isEqual(const PlotSettings &ps) const
{
    return (
                (ps.numXTicks == numXTicks) &&
                (ps.numYTicks == numYTicks) &&
                AreTheSame(ps.minX, minX) &&
                AreTheSame(ps.maxX, maxX) &&
                AreTheSame(ps.minY, minY) &&
                AreTheSame(ps.maxY, maxY)
            );
}

void PlotSettings::scroll(int dx, int dy)
{
    double stepX = spanX() / numXTicks;
    minX += dx * stepX;
    maxX += dx * stepX;

    double stepY = spanY() / numYTicks;
    minY += dy * stepY;
    maxY += dy * stepY;
}

void PlotSettings::adjust()
{
    adjustAxis(minX, maxX, numXTicks);
    adjustAxis(minY, maxY, numYTicks);
}

void PlotSettings::adjustAxis(double &min, double &max, int &numTicks)
{
    const int MinTicks = 4;
    double grossStep = (max - min) / MinTicks;
    double step = pow(10.0, floor(log10(grossStep)));

    if (5 * step < grossStep) {
        step *= 5;
    } else if (2 * step < grossStep) {
        step *= 2;
    }

    numTicks = int(ceil(max / step) - floor(min / step));
    if (numTicks < MinTicks)
        numTicks = MinTicks;
    min = floor(min / step) * step;
    max = ceil(max / step) * step;
}
