#ifndef SORRY_BACKEND_H
#define SORRY_BACKEND_H

#include "Sorry-MCTS/sorry.hpp"
#include "Sorry-MCTS/sorryMcts.hpp"

#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QObject>
#include <QQmlEngine>
#include <QVector>

#include <random>
#include <thread>
#include <vector>

class ActionForQml : public QObject {
  Q_OBJECT
  QML_ELEMENT
  Q_PROPERTY(QString name READ name NOTIFY nameChanged)
  Q_PROPERTY(double score READ score NOTIFY scoreChanged)
public:
  ActionForQml() = default;
  ActionForQml(const sorry::Action &action, double score) : action_(action), score_(score) {}

  void updateScore(double newScore) {
    score_ = newScore;
    emit scoreChanged();
  }

  QString name() const { return QString::fromStdString(action_.toString()); }
  Q_INVOKABLE const sorry::Action& getAction() const { return action_; }
  double score() const { return score_; }

signals:
  void nameChanged();
  void scoreChanged();

private:
  sorry::Action action_;
  double score_;
};

class ActionsList : public QAbstractListModel {
public:
  enum ActionRoles {
    NameRole = Qt::UserRole + 1,
    ScoreRole,
    AverageMovesRole
  };
  ActionsList();
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void setActionsAndScores(const std::vector<ActionScore> &actionsAndScores);
  std::optional<sorry::Action> getAction(int index) const;
  void reset();
protected:
  QHash<int, QByteArray> roleNames() const override {
    QHash<int, QByteArray> roles;
    roles[NameRole] = "name";
    roles[ScoreRole] = "score";
    roles[AverageMovesRole] = "averageMoves";
    return roles;
  }
private:
  QVector<ActionScore> actions_;
};

class SorryBackend : public QObject {
  Q_OBJECT
  QML_ELEMENT
  QML_SINGLETON
  Q_PROPERTY(QStringList dataList READ dataList NOTIFY dataListChanged)
  Q_PROPERTY(ActionsList* actionListModel READ actionListModel NOTIFY actionListModelChanged)
  Q_PROPERTY(int randomSeed READ randomSeed CONSTANT)
  Q_PROPERTY(int moveCount READ moveCount NOTIFY moveCountChanged)
public:
  explicit SorryBackend(QObject *parent = nullptr);
  ~SorryBackend();
  Q_INVOKABLE void test();
  QStringList dataList() { return m_dataList; }
  int randomSeed() const { return randomSeed_; }
  int moveCount() const { return sorryState_.getTotalActionCount(); }
  Q_INVOKABLE QVector<ActionForQml*> getActions();
  Q_INVOKABLE void doAction(int index);
  Q_INVOKABLE QVector<int> getPiecePositions() const;
  Q_INVOKABLE QVector<QString> getCardStrings() const;
  Q_INVOKABLE QVector<int> getCardIndicesForAction(int index) const;
  Q_INVOKABLE QVector<int> getSrcAndDestPositionsForAction(int index) const;
  ActionsList* actionListModel();

signals:
  void dataListChanged();
  void actionsChanged();
  void actionListModelChanged();
  void actionScoresChanged(std::vector<ActionScore> actionScores);
  void moveCountChanged();

private:
  SorryMcts mcts_{20};
  ExplicitTerminator mctsTerminator_;
  std::thread actionProberThread_;
  std::atomic<bool> runProber_;

  std::thread mctsThread_;
  int randomSeed_;
  std::mt19937 eng_;
  sorry::Sorry sorryState_;
  std::mutex actionsMutex_;
  QVector<ActionForQml*> actions_;
  QStringList m_dataList;
  ActionsList actionsList_;

  void calculateScores();
  void probeActions();
  void terminateThreads();
};

#endif // SORRY_BACKEND_H
