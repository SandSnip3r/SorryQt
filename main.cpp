#include "sorry_backend.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>

int main(int argc, char *argv[]) {
  QGuiApplication app(argc, argv);

  QQmlApplicationEngine engine;
  const QUrl url(QStringLiteral("qrc:/Sorry/Main.qml"));
  QObject::connect(
      &engine,
      &QQmlApplicationEngine::objectCreationFailed,
      &app,
      []() { QCoreApplication::exit(-1); },
      Qt::QueuedConnection);
  engine.load(url);

  auto *sorry = engine.singletonInstance<SorryBackend*>("Sorry","SorryBackend");
  
  // TODO: Initialize sorry state, temporarily.
  sorry->test();

  return app.exec();
}
