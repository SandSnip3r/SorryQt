#ifndef SORRY_BACKEND_H
#define SORRY_BACKEND_H

#include <sorry/agent/base/baseAgent.hpp>
#include <sorry/agent/mcts/sorryMcts.hpp>
#include <sorry/engine/sorry.hpp>

#pragma push_macro("slots")
#undef slots
#include <Python.h>
#pragma pop_macro("slots")

#include <QAbstractItemModel>
#include <QAbstractListModel>
#include <QList>
#include <QObject>
#include <QQmlEngine>
#include <QVector>

#include <map>
#include <random>
#include <thread>
#include <vector>

class PlayerType : public QObject {
  Q_OBJECT
  QML_ELEMENT
public:
  explicit PlayerType() = default;
  enum PlayerTypeEnum {
    Human,
    Mcts,
    MctsAssistedHuman,
    Rl
  };
  Q_ENUM(PlayerTypeEnum);
};

class PlayerColor : public QObject {
  Q_OBJECT
  QML_ELEMENT
public:
  explicit PlayerColor() = default;
  enum PlayerColorEnum {
    Green,
    Red,
    Blue,
    Yellow,
    GameOver
  };
  Q_ENUM(PlayerColorEnum);
};

class MoveForArrow : public QObject {
  Q_OBJECT
  QML_ELEMENT
  Q_PROPERTY(PlayerColor::PlayerColorEnum playerColor READ playerColor CONSTANT)
  Q_PROPERTY(int pieceIndex READ pieceIndex CONSTANT)
  Q_PROPERTY(int srcPosition READ srcPosition CONSTANT)
  Q_PROPERTY(int destPosition READ destPosition CONSTANT)
public:
  explicit MoveForArrow(QObject *parent = nullptr) : QObject(parent) {}
  MoveForArrow(PlayerColor::PlayerColorEnum playerColor, int pieceIndex, int srcPosition, int destPosition) :
      playerColor_(playerColor), pieceIndex_(pieceIndex), srcPosition_(srcPosition), destPosition_(destPosition) {}
  ~MoveForArrow() {}
  PlayerColor::PlayerColorEnum playerColor() const { return playerColor_; };
  int pieceIndex() const { return pieceIndex_; };
  int srcPosition() const { return srcPosition_; };
  int destPosition() const { return destPosition_; };
private:
  const PlayerColor::PlayerColorEnum playerColor_{PlayerColor::PlayerColorEnum::Green};
  const int pieceIndex_{0};
  const int srcPosition_{0};
  const int destPosition_{0};
};

class ActionForQml : public QObject {
  Q_OBJECT
  QML_ELEMENT
  Q_PROPERTY(QString name READ name NOTIFY nameChanged)
  Q_PROPERTY(double score READ score NOTIFY scoreChanged)
public:
  ActionForQml() = default;
  ActionForQml(const sorry::engine::Action &action, double score) : action_(action), score_(score) {}

  void updateScore(double newScore) {
    score_ = newScore;
    emit scoreChanged();
  }

  QString name() const { return QString::fromStdString(action_.toString()); }
  Q_INVOKABLE const sorry::engine::Action& getAction() const { return action_; }
  double score() const { return score_; }

signals:
  void nameChanged();
  void scoreChanged();

private:
  sorry::engine::Action action_;
  double score_;
};

class ActionsList : public QAbstractListModel {
public:
  enum ActionRoles {
    NameRole = Qt::UserRole + 1,
    ScoreRole,
    IsBestRole
  };
  ActionsList();
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void setActionsAndScores(const std::vector<sorry::agent::ActionScore> &actionsAndScores);
  std::optional<sorry::engine::Action> getAction(int index) const;
protected:
  QHash<int, QByteArray> roleNames() const override {
    QHash<int, QByteArray> roles;
    roles[NameRole] = "name";
    roles[ScoreRole] = "score";
    roles[IsBestRole] = "isBest";
    return roles;
  }
private:
  std::vector<sorry::agent::ActionScore> actionScores_;
  mutable std::recursive_mutex mutex_;
  size_t bestIndex_{0};
};

class SorryBackend : public QObject {
  Q_OBJECT
  QML_ELEMENT
  QML_SINGLETON
  Q_PROPERTY(ActionsList* actionListModel READ actionListModel NOTIFY actionListModelChanged)
  Q_PROPERTY(int randomSeed READ randomSeed NOTIFY randomSeedChanged)
  Q_PROPERTY(int faceDownCardsCount READ faceDownCardsCount NOTIFY boardStateChanged)
  Q_PROPERTY(int iterationCount READ iterationCount NOTIFY iterationCountChanged)
  Q_PROPERTY(PlayerColor::PlayerColorEnum playerTurn READ playerTurn NOTIFY playerTurnChanged)
  Q_PROPERTY(PlayerType::PlayerTypeEnum playerType READ playerType NOTIFY playerTurnChanged)
  Q_PROPERTY(QString winner READ winner NOTIFY winnerChanged)
public:
  explicit SorryBackend(QObject *parent = nullptr);
  ~SorryBackend();

  Q_INVOKABLE void resetGame();
  int randomSeed() const { return randomSeed_; }
  int faceDownCardsCount() const;
  PlayerColor::PlayerColorEnum playerTurn() const;
  PlayerType::PlayerTypeEnum playerType() const;
  Q_INVOKABLE void doActionFromActionList(int index);
  Q_INVOKABLE QList<PlayerColor::PlayerColorEnum> getPlayers() const;
  Q_INVOKABLE QList<int> getPiecePositionsForPlayer(PlayerColor::PlayerColorEnum playerColor) const;
  Q_INVOKABLE QList<QString> getCardStringsForPlayer(PlayerColor::PlayerColorEnum playerColor) const;
  Q_INVOKABLE PlayerColor::PlayerColorEnum getPlayerForAction(int index) const;
  Q_INVOKABLE QList<int> getCardIndicesForAction(int index) const;
  Q_INVOKABLE QList<MoveForArrow*> getMovesForAction(int index) const;
  ActionsList* actionListModel();
  int iterationCount() const;
  QString winner() const;

signals:
  void boardStateChanged();
  void actionListModelChanged();
  void actionScoresChanged(std::vector<sorry::agent::ActionScore> actionScores);
  void winRatesChanged(std::vector<double> winRates);
  void iterationCountChanged();
  void randomSeedChanged();
  void faceDownCardsCountChanged();
  void playerTurnChanged();
  void actionChosen(sorry::engine::Action action);
  void winnerChanged();

private:
  bool hiddenHand_{false};
  static constexpr bool kHumanIsMctsAssisted{false};
  std::map<sorry::engine::PlayerColor, PlayerType::PlayerTypeEnum> playerTypes_;
  SorryMcts mcts_{2};
  sorry::agent::BaseAgent *rlAgent_{nullptr};
  ExplicitTerminator mctsTerminator_;
  std::thread actionProberThread_;
  std::atomic<bool> runProber_;

  std::thread mctsThread_;

  PyThreadState *savedThreadState_;
  std::thread reinforceThread_;
  int randomSeed_;
  std::mt19937 eng_;
  sorry::engine::Sorry sorryState_{sorry::engine::PlayerColor::kGreen};
  std::mutex actionsMutex_;
  QVector<ActionForQml*> actions_;
  ActionsList actionsList_;
  int lastIterationCount_{0};

  void updateAi();
  void initializeActions();
  void initializeGame();
  void runMctsAssistant();
  void probeActions();
  void terminateThreads();
  void runMctsAgent();
  void runRlAgent();
  void doActionAsAgent(const sorry::engine::Action &action);
  void doAction(const sorry::engine::Action &action);

  static PlayerColor::PlayerColorEnum sorryEnumToBackendEnum(sorry::engine::PlayerColor playerColor);
  static sorry::engine::PlayerColor backendEnumToSorryEnum(PlayerColor::PlayerColorEnum playerColor);
};

#endif // SORRY_BACKEND_H
