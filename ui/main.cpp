#include "sorry_backend.h"

#include <QGuiApplication>
#include <QQmlApplicationEngine>

#pragma push_macro("slots")
#undef slots
#include <pybind11/embed.h>
#pragma pop_macro("slots")

int main(int argc, char *argv[]) {
  pybind11::scoped_interpreter guard;
  QGuiApplication app(argc, argv);

  QQmlApplicationEngine engine;
  const QUrl url(QStringLiteral("qrc:/qt/qml/Sorry/Main.qml"));
  QObject::connect(
      &engine,
      &QQmlApplicationEngine::objectCreationFailed,
      &app,
      []() { QCoreApplication::exit(-1); },
      Qt::QueuedConnection);
  engine.load(url);

#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
  SorryBackend *sorryBackend = engine.singletonInstance<SorryBackend*>("Sorry","SorryBackend");
#else
  // Get the type ID
  int sorryBackendTypeId = qMetaTypeId<SorryBackend*>();

  // Obtain the singleton instance
  SorryBackend *sorryBackend = engine.singletonInstance<SorryBackend*>(sorryBackendTypeId);
#endif

  return app.exec();
}
