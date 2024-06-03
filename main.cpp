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

#if QT_VERSION >= QT_VERSION_CHECK(6, 5, 0)
  auto *sorry = engine.singletonInstance<SorryBackend*>("Sorry","SorryBackend");
#else
  // Register the singleton type
  qmlRegisterSingletonType<SorryBackend>("Sorry", 1, 0, "SorryBackend", [](QQmlEngine *engine, QJSEngine *scriptEngine) -> QObject* {
      Q_UNUSED(engine)
      Q_UNUSED(scriptEngine)
      return new SorryBackend();
  });

  // Get the type ID
  int sorryBackendTypeId = qMetaTypeId<SorryBackend*>();

  // Obtain the singleton instance
  auto *sorry = engine.singletonInstance<SorryBackend*>(sorryBackendTypeId);
#endif

  return app.exec();
}
