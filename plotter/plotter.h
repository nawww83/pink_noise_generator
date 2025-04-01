#pragma once

#include <QWidget>
#include <QMap>
#include <QPixmap>
#include <QVector>

class QToolButton;
class PlotSettings;

class Plotter : public QWidget
{
    Q_OBJECT
public:
    explicit Plotter(QWidget *parent = nullptr);

    void setPlotSettings(const PlotSettings &settings);

    void setCurveData(int id, const QVector<QPointF> &data);

    void clearCurve(int id);

    void clearCurves();

    QSize minimumSizeHint() const;

    QSize sizeHint() const;

public slots:
    void zoomIn();

    void zoomOut();

protected:
    void paintEvent(QPaintEvent *event);

    void resizeEvent(QResizeEvent *event);

    void mousePressEvent(QMouseEvent *event);

    void mouseMoveEvent(QMouseEvent *event);

    void mouseReleaseEvent(QMouseEvent *event);

    void keyPressEvent(QKeyEvent *event);

    void wheelEvent(QWheelEvent *event);

private:
    void updateRubberBandRegion();

    void refreshPixmap();

    void drawGrid(QPainter *painter);

    void drawCurves(QPainter *painter);

    enum { Margin = 50 };

    QToolButton *zoomInButton;

    QToolButton *zoomOutButton;

    QMap<int, QVector<QPointF> > curveMap;

    QVector<PlotSettings> zoomStack;

    int curZoom;

    bool rubberBandIsShown;

    QRect rubberBandRect;

    QPixmap pixmap;
};

class PlotSettings
{
public:
    explicit PlotSettings();

    bool isEqual(const PlotSettings& ps) const;

    void scroll(int dx, int dy);

    void adjust();

    double spanX() const { return maxX - minX; }

    double spanY() const { return maxY - minY; }

    double minX;

    double maxX;

    int numXTicks;

    double minY;

    double maxY;

    int numYTicks;

private:
    static void adjustAxis(double &min, double &max, int &numTicks);
};
