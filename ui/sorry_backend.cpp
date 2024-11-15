#include "sorry_backend.h"

// Python & Qt both define `slots`
//  https://stackoverflow.com/a/49359288
#pragma push_macro("slots")
#undef slots
#include <pybind11/embed.h>
#include <pybind11/gil.h>

#include <sorry/agent/rl/reinforceAgent.hpp>
#pragma pop_macro("slots")

#include <iostream>

SorryBackend::SorryBackend(QObject *parent) : QObject(parent) {
  connect(this, &SorryBackend::actionScoresChanged, &actionsList_, &ActionsList::setActionsAndScores);
  connect(this, &SorryBackend::actionChosen, this, &SorryBackend::doActionAsAgent);
  initializeGame();
}

// Construct a player type for each player color.

// At the start of a player's turn:
// 1. Start computing action preferences
// 2. Display action preferences as they update

// Human players have no logic to execute and no preferences to display.
// RL player has just a quick bit of logic to execute, preferences are immediately available.
// MCTS player has a lot of logic to execute, preferences are not immediately available; async probe them.
// MCTS assisted human is the same as mcts, except the human makes the final choice and mcts runs indefinitely.

SorryBackend::~SorryBackend() {
  terminateThreads();
}

void SorryBackend::initializeGame() {
  // Initialize random engine
  randomSeed_ = std::random_device()();
  std::cout << "Seed: " << randomSeed_ << std::endl;
  emit randomSeedChanged();
  eng_ = std::mt19937(randomSeed_);

  // Create new Sorry game
  sorryState_ = sorry::engine::Sorry({sorry::engine::PlayerColor::kGreen, sorry::engine::PlayerColor::kBlue});
  // sorryState_ = sorry::engine::Sorry({sorry::engine::PlayerColor::kGreen, sorry::engine::PlayerColor::kRed});
  // sorryState_ = sorry::engine::Sorry({sorry::engine::PlayerColor::kGreen, sorry::engine::PlayerColor::kRed, sorry::engine::PlayerColor::kBlue, sorry::engine::PlayerColor::kYellow});
  sorryState_.reset(eng_);

  rlAgent_ = new sorry::agent::ReinforceAgent();
  rlAgent_->seed(randomSeed_);

  playerTypes_[sorry::engine::PlayerColor::kGreen] = PlayerType::Rl;
  playerTypes_[sorry::engine::PlayerColor::kBlue] = PlayerType::Human;
  // playerTypes_[sorry::engine::PlayerColor::kGreen] = PlayerType::Mcts;
  // playerTypes_[sorry::engine::PlayerColor::kGreen] = PlayerType::Human;
  // playerTypes_[sorry::engine::PlayerColor::kRed] = PlayerType::Mcts;
  // playerTypes_[sorry::engine::PlayerColor::kBlue] = PlayerType::Mcts;
  // playerTypes_[sorry::engine::PlayerColor::kYellow] = PlayerType::Mcts;

  // If no player is human, disable hidden hand
  if (hiddenHand_ == true) {
    bool noneAreHuman{true};
    for (const auto &playerType : playerTypes_) {
      if (playerType.second == PlayerType::Human || playerType.second == PlayerType::MctsAssistedHuman) {
        noneAreHuman = false;
        break;
      }
    }
    if (noneAreHuman) {
      std::cout << "No player is human, disabling hidden hand" << std::endl;
    }
    hiddenHand_ = !noneAreHuman;
  }

  // Emit signals
  emit winnerChanged();
  emit playerTurnChanged();
  emit boardStateChanged();
  initializeActions();
  updateAi();
}

void SorryBackend::updateAi() {
  const auto currentPlayerTurn = sorryState_.getPlayerTurn();
  const auto playerType = playerTypes_.at(currentPlayerTurn);
  if (playerType == PlayerType::Mcts) {
    // Player is a bot. Start mcts agent.
    runMctsAgent();
  } else if (playerType == PlayerType::Human) {
    // Nothing to do.
  } else if (playerType == PlayerType::MctsAssistedHuman) {
    // Run assistive mcts.
    runMctsAssistant();
  } else if (playerType == PlayerType::Rl) {
    runRlAgent();
  } else if (playerType == PlayerType::RlAssistedHuman) {
    runRlAgentAssistant();
  }
}

void SorryBackend::resetGame() {
  terminateThreads();
  initializeGame();
}

int SorryBackend::faceDownCardsCount() const {
  return sorryState_.getFaceDownCardsCount();
}

PlayerColor::PlayerColorEnum SorryBackend::playerTurn() const {
  if (sorryState_.gameDone()) {
    return PlayerColor::GameOver;
  }
  return sorryEnumToBackendEnum(sorryState_.getPlayerTurn());
}

PlayerType::PlayerTypeEnum SorryBackend::playerType() const {
  if (sorryState_.gameDone()) {
    return PlayerType::Human; // TODO
  }
  return playerTypes_.at(sorryState_.getPlayerTurn());
}

int SorryBackend::iterationCount() const {
  return lastIterationCount_;
}

QString SorryBackend::winner() const {
  if (!sorryState_.gameDone()) {
    return tr("");
  }
  return QString::fromStdString(std::string(sorry::engine::toString(sorryState_.getWinner())));
}

void SorryBackend::probeActions() {
  runProber_ = true;
  while (runProber_) {
    const std::vector<sorry::agent::ActionScore> actionsAndScores = mcts_.getActionScores();
    emit actionScoresChanged(actionsAndScores);
    const auto winRates = mcts_.getWinRates();
    emit winRatesChanged(winRates);
    lastIterationCount_ = mcts_.getIterationCount();
    emit iterationCountChanged();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void SorryBackend::runMctsAssistant() {
  if (sorryState_.getActions().size() <= 1) {
    return;
  }
  // Start running MCTS to figure out the value of each action and display it to the human player as a helpful hint.
  mctsTerminator_.setDone(false);
  mctsThread_ = std::thread([&](){
    mcts_.run(sorryState_, &mctsTerminator_);
    // MCTS is terminated when the user selects an action. Upon returning from run(), we can immediately reset MCTS.
    mcts_.reset();
  });
  if (!hiddenHand_) {
    // Start another thread to periodically get data from mcts.
    actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
  }
}

void SorryBackend::runMctsAgent() {
  // Start running MCTS to figure out the value of each action.
  mctsThread_ = std::thread([&](){
    // mcts_.run(sorryState_, std::chrono::seconds(5));
    mcts_.run(sorryState_, 50000);
    const sorry::engine::Action bestAction = mcts_.pickBestAction();
    // Terminate prober
    runProber_ = false;
    if (actionProberThread_.joinable()) {
      actionProberThread_.join();
    }

    mcts_.reset();
    emit actionChosen(bestAction);
  });
  // Start another thread to periodically get data from mcts.
  actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
}

void SorryBackend::runRlAgent() {
  // Since we're running the RL agent in another thread, we need to do some manual GIL management.
  //  https://docs.python.org/3/c-api/init.html
  savedThreadState_ = PyEval_SaveThread();
  reinforceThread_ = std::thread([&](){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    rlAgent_->run(sorryState_);
    const std::vector<sorry::agent::ActionScore> actionsAndScores = rlAgent_->getActionScores();
    if (!hiddenHand_) {
      emit actionScoresChanged(actionsAndScores);
      std::this_thread::sleep_for(std::chrono::milliseconds(800));
    }
    const sorry::engine::Action bestAction = rlAgent_->pickBestAction();
    emit actionChosen(bestAction);
    PyGILState_Release(gstate);
    PyEval_RestoreThread(savedThreadState_);
  });
}

void SorryBackend::runRlAgentAssistant() {
  rlAgent_->run(sorryState_);
  const std::vector<sorry::agent::ActionScore> actionsAndScores = rlAgent_->getActionScores();
  emit actionScoresChanged(actionsAndScores);
}

void SorryBackend::terminateThreads() {
  // Terminate prober
  runProber_ = false;
  if (actionProberThread_.joinable()) {
    actionProberThread_.join();
  }
  // Terminate mcts
  mctsTerminator_.setDone(true);
  if (mctsThread_.joinable()) {
    mctsThread_.join();
  }
  // Terminate RL agent
  if (reinforceThread_.joinable()) {
    reinforceThread_.join();
  }
}

PlayerColor::PlayerColorEnum SorryBackend::sorryEnumToBackendEnum(sorry::engine::PlayerColor playerColor) {
  if (playerColor == sorry::engine::PlayerColor::kGreen) {
    return PlayerColor::Green;
  } else if (playerColor == sorry::engine::PlayerColor::kRed) {
    return PlayerColor::Red;
  } else if (playerColor == sorry::engine::PlayerColor::kBlue) {
    return PlayerColor::Blue;
  } else if (playerColor == sorry::engine::PlayerColor::kYellow) {
    return PlayerColor::Yellow;
  } else {
    throw std::runtime_error("Invalid player color");
  }
}

sorry::engine::PlayerColor SorryBackend::backendEnumToSorryEnum(PlayerColor::PlayerColorEnum playerColor) {
  if (playerColor == PlayerColor::Green) {
    return sorry::engine::PlayerColor::kGreen;
  } else if (playerColor == PlayerColor::Red) {
    return sorry::engine::PlayerColor::kRed;
  } else if (playerColor == PlayerColor::Blue) {
    return sorry::engine::PlayerColor::kBlue;
  } else if (playerColor == PlayerColor::Yellow) {
    return sorry::engine::PlayerColor::kYellow;
  } else {
    throw std::runtime_error("Invalid player color");
  }
}

void SorryBackend::doActionFromActionList(int index) {
  // Received a request from the UI to do an action from our action list.
  // Get the action as quickly as possible
  const std::optional<sorry::engine::Action> action = actionsList_.getAction(index);
  // We assume that the mcts assistant is running, kill it
  terminateThreads();
  // Do action
  if (!action) {
    std::cout << "Want to do unknown action (index " << index << ")" << std::endl;
    return;
  }
  doAction(*action);
}

void SorryBackend::doActionAsAgent(const sorry::engine::Action &action) {
  terminateThreads();
  doAction(action);
}

void SorryBackend::doAction(const sorry::engine::Action &action) {
  std::cout << "Doing action " << action.toString() << std::endl;
  const auto prevPlayerTurn = sorryState_.getPlayerTurn();
  sorryState_.doAction(action, eng_);
  emit boardStateChanged();
  const auto currentPlayerTurn = sorryState_.getPlayerTurn();
  if (currentPlayerTurn != prevPlayerTurn) {
    emit playerTurnChanged();
  }
  initializeActions();
  if (!sorryState_.gameDone()) {
    updateAi();
  } else {
    emit playerTurnChanged();
    emit winnerChanged();
  }
}

void SorryBackend::initializeActions() {
  PlayerType::PlayerTypeEnum currentPlayerType = playerTypes_.at(sorryState_.getPlayerTurn());
  if (hiddenHand_ && currentPlayerType != PlayerType::Human && currentPlayerType != PlayerType::MctsAssistedHuman) {
    actionScoresChanged({});
    return;
  }
  const auto actions = sorryState_.getActions();
  std::vector<sorry::agent::ActionScore> actionScores;
  actionScores.reserve(actions.size());
  for (const auto &action : actions) {
    actionScores.emplace_back(action, 0);
  }
  actionScoresChanged(actionScores);
}

QList<PlayerColor::PlayerColorEnum> SorryBackend::getPlayers() const {
  const auto players = sorryState_.getPlayers();
  QList<PlayerColor::PlayerColorEnum> result;
  for (const auto player : players) {
    result.emplace_back(sorryEnumToBackendEnum(player));
  }
  return result;
}

QList<int> SorryBackend::getPiecePositionsForPlayer(PlayerColor::PlayerColorEnum playerColor) const {
  const auto positions = sorryState_.getPiecePositionsForPlayer(backendEnumToSorryEnum(playerColor));
  QList<int> result;
  for (int i=0; i<4; ++i) {
    result.push_back(positions[i]);
  }
  return result;
}

QList<QString> SorryBackend::getCardStringsForPlayer(PlayerColor::PlayerColorEnum playerColor) const {
  if (hiddenHand_ && playerTypes_.at(backendEnumToSorryEnum(playerColor)) != PlayerType::Human && playerTypes_.at(backendEnumToSorryEnum(playerColor)) != PlayerType::MctsAssistedHuman) {
    return { QString::fromStdString(toString(sorry::engine::Card::kUnknown)),
             QString::fromStdString(toString(sorry::engine::Card::kUnknown)),
             QString::fromStdString(toString(sorry::engine::Card::kUnknown)),
             QString::fromStdString(toString(sorry::engine::Card::kUnknown)),
             QString::fromStdString(toString(sorry::engine::Card::kUnknown)) };
  }
  QList<QString> result;
  const auto hand = sorryState_.getHandForPlayer(backendEnumToSorryEnum(playerColor));
  for (const auto card : hand) {
    result.push_back(QString::fromStdString(toString(card)));
  }
  return result;
}

PlayerColor::PlayerColorEnum SorryBackend::getPlayerForAction(int index) const {
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to get player for unknown action (index " << index << ")" << std::endl;
    return {};
  }
  return sorryEnumToBackendEnum(action->playerColor);
}

QList<int> SorryBackend::getCardIndicesForAction(int index) const {
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to get card indices for unknown action (index " << index << ")" << std::endl;
    return {};
  }
  const auto card = action->card;
  const auto hand = sorryState_.getHandForPlayer(action->playerColor);
  QList<int> result;
  for (int i=0; i<hand.size(); ++i) {
    if (hand[i] == card) {
      result.push_back(i);
    }
  }
  return result;
}

QList<MoveForArrow*> SorryBackend::getMovesForAction(int index) const {
  QList<MoveForArrow*> result;
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to get src and dest pos for unknown action (index " << index << ")" << std::endl;
    return {};
  }
  const auto sorryMoves = sorryState_.getMovesForAction(*action);
  for (const auto &move : sorryMoves) {
    result.push_back(new MoveForArrow(sorryEnumToBackendEnum(move.playerColor), move.pieceIndex, move.srcPosition, move.destPosition));
  }
  return result;
}

ActionsList* SorryBackend::actionListModel() {
  return &actionsList_;
}

// --------------------------------------------------------------------------------------------------------------

ActionsList::ActionsList() {}

int ActionsList::rowCount(const QModelIndex &parent) const {
  std::unique_lock lock(mutex_);
  return actionScores_.size();
}

QVariant ActionsList::data(const QModelIndex &index, int role) const {
  if (!index.isValid()) {
    return QVariant();
  }
  std::unique_lock lock(mutex_);
  if (index.row() < 0 || index.row() >= actionScores_.size()) {
    return QVariant();
  }

  if (actionScores_.empty()) {
    return QVariant();
  }

  const sorry::agent::ActionScore &actionScore = actionScores_.at(index.row());
  if (role == NameRole) {
    const sorry::engine::Action &action = actionScore.action;
    if (action.actionType == sorry::engine::Action::ActionType::kDiscard) {
      return tr("Discard %1").arg(QString::fromStdString(sorry::engine::toString(action.card)));
    }
    return QVariant(QString::fromStdString(action.toString()));
  } else if (role == ScoreRole) {
    return QVariant(actionScore.score);
  } else if (role == IsBestRole) {
    return QVariant(index.row() == bestIndex_);
  }
  return QVariant();
}

void ActionsList::setActionsAndScores(const std::vector<sorry::agent::ActionScore> &actionsAndScores) {
  std::unique_lock lock(mutex_);
  std::vector<bool> actionSeen(actionScores_.size(), false);
  double bestScore = 0.0;
  size_t bestIndex{0};
  for (size_t i=0; i<actionsAndScores.size(); ++i) {
    if (actionsAndScores.at(i).score > bestScore) {
      bestScore = actionsAndScores.at(i).score;
      bestIndex = i;
    }
  }
  size_t lastBestIndex = bestIndex_;
  for (size_t newActionIndex=0; newActionIndex<actionsAndScores.size(); ++newActionIndex) {
    const auto &newActionScore = actionsAndScores.at(newActionIndex);
    const sorry::engine::Action &newAction = newActionScore.action;
    bool foundAction = false;
    for (size_t existingActionIndex=0; existingActionIndex<actionScores_.size(); ++existingActionIndex) {
      sorry::agent::ActionScore &actionScore = actionScores_.at(existingActionIndex);
      if (actionScore.action == newAction) {
        // Found our action, update the score
        actionScore.score = newActionScore.score;
        if (newActionIndex == bestIndex) {
          bestIndex_ = existingActionIndex;
        }
        emit dataChanged(this->index(existingActionIndex), this->index(existingActionIndex), {ScoreRole, IsBestRole});
        actionSeen[existingActionIndex] = true;
        foundAction = true;
        break;
      }
    }
    if (foundAction) {
      continue;
    }
    // New action not yet in our list.
    beginInsertRows(QModelIndex(), actionScores_.size(), actionScores_.size());
    actionScores_.push_back(newActionScore);
    if (newActionIndex == bestIndex) {
      bestIndex_ = actionScores_.size()-1;
    }
    endInsertRows();
  }
  for (int i=actionSeen.size()-1; i>=0; --i) {
    if (!actionSeen[i]) {
      beginRemoveRows(QModelIndex(), i, i);
      actionScores_.erase(actionScores_.begin() + i);
      if (bestIndex_ >= i) {
        --bestIndex_;
      }
      endRemoveRows();
    }
  }
  if (bestIndex_ != lastBestIndex) {
    emit dataChanged(this->index(bestIndex_), this->index(bestIndex_), {IsBestRole});
    emit dataChanged(this->index(lastBestIndex), this->index(lastBestIndex), {IsBestRole});
  }
}

std::optional<sorry::engine::Action> ActionsList::getAction(int index) const {
  std::unique_lock lock(mutex_);
  if (index < 0 || index >= actionScores_.size()) {
    return {};
  }
  return actionScores_.at(index).action;
}
