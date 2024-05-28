#ifndef SORRY_BACKEND_H
#define SORRY_BACKEND_H

#include "Sorry-MCTS/sorry.hpp"
#include "Sorry-MCTS/sorryMcts.hpp"

#include <QAbstractItemModel>
#include <QObject>
#include <QQmlEngine>
#include <QVector>

#include <random>
#include <thread>
#include <vector>

class ActionForQml : public QObject {
  Q_OBJECT
  Q_PROPERTY(QString name READ name CONSTANT)
  Q_PROPERTY(double score READ score CONSTANT)
public:
  ActionForQml() = default;
  ActionForQml(const sorry::Action &action, double score) : action_(action), score_(score) {}

  QString name() const { return QString::fromStdString(action_.toString()); }
  Q_INVOKABLE const sorry::Action& getAction() const { return action_; }
  double score() const { return score_; }

private:
  sorry::Action action_;
  double score_;
};

class SorryBackend : public QObject {
  Q_OBJECT
  QML_ELEMENT
  QML_SINGLETON
  Q_PROPERTY(QStringList dataList READ dataList NOTIFY dataListChanged)
public:
  explicit SorryBackend(QObject *parent = nullptr);
  Q_INVOKABLE void test();
  QStringList dataList() { return m_dataList; }
  Q_INVOKABLE QVector<ActionForQml*> getActions();
  Q_INVOKABLE void doAction(const ActionForQml *action);
  Q_INVOKABLE QVector<int> getPiecePositions() const;
  Q_INVOKABLE QVector<QString> getCardStrings() const;
  Q_INVOKABLE int getMoveCount() const;
  Q_INVOKABLE QVector<int> getCardIndicesForAction(const ActionForQml *qmlAction) const;
  Q_INVOKABLE QVector<int> getSrcAndDestPositionsForAction(const ActionForQml *qmlAction) const;

signals:
  void dataListChanged();
  void actionsChanged();

private:
  SorryMcts mcts_{20};
  ExplicitTerminator mctsTerminator_;
  std::thread actionProberThread_;
  std::atomic<bool> runProber_;

  std::thread mctsThread_;
  std::mt19937 eng_;
  sorry::Sorry sorryState_;
  std::mutex actionsMutex_;
  QVector<ActionForQml*> actions_;
  QStringList m_dataList;

  void calculateScores();
  void probeActions();
};

#endif // SORRY_BACKEND_H
