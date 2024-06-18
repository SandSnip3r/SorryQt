#ifndef SORRY_BACKEND_H
#define SORRY_BACKEND_H

#include "Sorry-MCTS/sorry.hpp"
#include "Sorry-MCTS/sorryMcts.hpp"

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
    MctsAssistedHuman
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
  MoveForArrow(PlayerColor::PlayerColorEnum playerColor, int pieceIndex, int srcPosition, int destPosition) :
      playerColor_(playerColor), pieceIndex_(pieceIndex), srcPosition_(srcPosition), destPosition_(destPosition) {}
  ~MoveForArrow() {}
  PlayerColor::PlayerColorEnum playerColor() const { return playerColor_; };
  int pieceIndex() const { return pieceIndex_; };
  int srcPosition() const { return srcPosition_; };
  int destPosition() const { return destPosition_; };
private:
  const PlayerColor::PlayerColorEnum playerColor_;
  const int pieceIndex_;
  const int srcPosition_;
  const int destPosition_;
};

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
    IsBestRole
  };
  ActionsList();
  int rowCount(const QModelIndex &parent = QModelIndex()) const override;
  QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
  void setActionsAndScores(const std::vector<ActionScore> &actionsAndScores);
  std::optional<sorry::Action> getAction(int index) const;
protected:
  QHash<int, QByteArray> roleNames() const override {
    QHash<int, QByteArray> roles;
    roles[NameRole] = "name";
    roles[ScoreRole] = "score";
    roles[IsBestRole] = "isBest";
    return roles;
  }
private:
  std::vector<ActionScore> actionScores_;
  mutable std::recursive_mutex mutex_;
  size_t bestIndex_{0};
};

class SorryBackend : public QObject {
  Q_OBJECT
  QML_ELEMENT
  QML_SINGLETON
  Q_PROPERTY(ActionsList* actionListModel READ actionListModel NOTIFY actionListModelChanged)
  Q_PROPERTY(int randomSeed READ randomSeed NOTIFY randomSeedChanged)
  Q_PROPERTY(int moveCount READ moveCount NOTIFY moveCountChanged)
  Q_PROPERTY(int iterationCount READ iterationCount NOTIFY iterationCountChanged)
  Q_PROPERTY(PlayerColor::PlayerColorEnum playerTurn READ playerTurn NOTIFY playerTurnChanged)
  Q_PROPERTY(PlayerType::PlayerTypeEnum playerType READ playerType NOTIFY playerTurnChanged)
public:
  explicit SorryBackend(QObject *parent = nullptr);
  ~SorryBackend();

  Q_INVOKABLE void resetGame();
  int randomSeed() const { return randomSeed_; }
  int moveCount() const { return 0; /* sorryState_.getTotalActionCount(); */ }
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

signals:
  void boardStateChanged();
  void actionListModelChanged();
  void actionScoresChanged(std::vector<ActionScore> actionScores);
  void winRatesChanged(std::vector<double> winRates);
  void iterationCountChanged();
  void randomSeedChanged();
  void moveCountChanged();
  void playerTurnChanged();
  void actionChosen(sorry::Action action);

private:
  static constexpr bool kHiddenHand{true};
  static constexpr bool kHumanIsMctsAssisted{false};
  std::map<sorry::PlayerColor, PlayerType::PlayerTypeEnum> playerTypes_;
  SorryMcts mcts_{2};
  ExplicitTerminator mctsTerminator_;
  std::thread actionProberThread_;
  std::atomic<bool> runProber_;

  std::thread mctsThread_;
  int randomSeed_;
  std::mt19937 eng_;
  sorry::Sorry sorryState_{sorry::PlayerColor::kGreen};
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
  void doActionAsAgent(const sorry::Action &action);
  void doAction(const sorry::Action &action);

  static PlayerColor::PlayerColorEnum sorryEnumToBackendEnum(sorry::PlayerColor playerColor);
  static sorry::PlayerColor backendEnumToSorryEnum(PlayerColor::PlayerColorEnum playerColor);
};

#endif // SORRY_BACKEND_H
