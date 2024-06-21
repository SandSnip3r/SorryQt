#include "sorry_backend.h"

#include <iostream>

SorryBackend::SorryBackend(QObject *parent) : QObject(parent) {
  connect(this, &SorryBackend::actionScoresChanged, &actionsList_, &ActionsList::setActionsAndScores);
  connect(this, &SorryBackend::actionChosen, this, &SorryBackend::doActionAsAgent);
  initializeGame();
}

SorryBackend::~SorryBackend() {
  terminateThreads();
}

void SorryBackend::initializeGame() {
  // Initialize random engine
  randomSeed_ = -365113758;
  // randomSeed_ = std::random_device()();
  emit randomSeedChanged();
  eng_ = std::mt19937(randomSeed_);

  // Create new Sorry game
  sorryState_ = sorry::Sorry({sorry::PlayerColor::kGreen, sorry::PlayerColor::kBlue});
  // sorryState_ = sorry::Sorry({sorry::PlayerColor::kGreen, sorry::PlayerColor::kRed, sorry::PlayerColor::kBlue, sorry::PlayerColor::kYellow});
  sorryState_.drawRandomStartingCards(eng_);
  std::cout << sorryState_.toString() << std::endl;

  playerTypes_[sorry::PlayerColor::kGreen] = PlayerType::MctsAssistedHuman;
  // playerTypes_[sorry::PlayerColor::kRed] = PlayerType::Mcts;
  playerTypes_[sorry::PlayerColor::kBlue] = PlayerType::Mcts;
  // playerTypes_[sorry::PlayerColor::kYellow] = PlayerType::Mcts;

  // If no player is human, disable hidden hand
  if (hiddenHand_ == true) {
    bool noneAreHuman{true};
    for (const auto &playerType : playerTypes_) {
      if (playerType.second == PlayerType::Human || playerType.second == PlayerType::MctsAssistedHuman) {
        noneAreHuman = false;
        break;
      }
    }
    hiddenHand_ = !noneAreHuman;
  }

  // Emit signals
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

void SorryBackend::probeActions() {
  runProber_ = true;
  while (runProber_) {
    const auto actionsAndScores = mcts_.getActionScores();
    std::cout << actionsAndScores.size() << " actionScores";
    for (const auto &actionScore : actionsAndScores) {
      std::cout << ", " << actionScore.action.toString() << "; " << actionScore.score;
    }
    std::cout << std::endl;
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
  // Start another thread to periodically get data from mcts.
  actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
}

void SorryBackend::runMctsAgent() {
  // Start running MCTS to figure out the value of each action.
  mctsThread_ = std::thread([&](){
    // mcts_.run(sorryState_, std::chrono::seconds(5));
    mcts_.run(sorryState_, 5000);
    const auto bestAction = mcts_.pickBestAction();
    // Terminate prober
    runProber_ = false;
    if (actionProberThread_.joinable()) {
      actionProberThread_.join();
    }

    mcts_.reset();
    emit actionChosen(bestAction);
  });
  if (!hiddenHand_) {
    // Start another thread to periodically get data from mcts.
    actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
  }
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
}

PlayerColor::PlayerColorEnum SorryBackend::sorryEnumToBackendEnum(sorry::PlayerColor playerColor) {
  if (playerColor == sorry::PlayerColor::kGreen) {
    return PlayerColor::Green;
  } else if (playerColor == sorry::PlayerColor::kRed) {
    return PlayerColor::Red;
  } else if (playerColor == sorry::PlayerColor::kBlue) {
    return PlayerColor::Blue;
  } else if (playerColor == sorry::PlayerColor::kYellow) {
    return PlayerColor::Yellow;
  } else {
    throw std::runtime_error("Invalid player color");
  }
}

sorry::PlayerColor SorryBackend::backendEnumToSorryEnum(PlayerColor::PlayerColorEnum playerColor) {
  if (playerColor == PlayerColor::Green) {
    return sorry::PlayerColor::kGreen;
  } else if (playerColor == PlayerColor::Red) {
    return sorry::PlayerColor::kRed;
  } else if (playerColor == PlayerColor::Blue) {
    return sorry::PlayerColor::kBlue;
  } else if (playerColor == PlayerColor::Yellow) {
    return sorry::PlayerColor::kYellow;
  } else {
    throw std::runtime_error("Invalid player color");
  }
}

void SorryBackend::doActionFromActionList(int index) {
  // Received a request from the UI to do an action from our action list.
  // Get the action as quickly as possible
  const auto action = actionsList_.getAction(index);
  // We assume that the mcts assistant is running, kill it
  terminateThreads();
  // Do action
  if (!action) {
    std::cout << "Want to do unknown action (index " << index << ")" << std::endl;
    return;
  }
  doAction(*action);
}

void SorryBackend::doActionAsAgent(const sorry::Action &action) {
  terminateThreads();
  doAction(action);
}

void SorryBackend::doAction(const sorry::Action &action) {
  std::cout << "Doing action " << action.toString() << std::endl;
  const auto prevPlayerTurn = sorryState_.getPlayerTurn();
  sorryState_.doAction(action, eng_);
  const auto currentPlayerTurn = sorryState_.getPlayerTurn();
  if (currentPlayerTurn != prevPlayerTurn) {
    emit playerTurnChanged();
  }
  emit boardStateChanged();
  if (!sorryState_.gameDone()) {
    initializeActions();
    updateAi();
  }
}

void SorryBackend::initializeActions() {
  if (hiddenHand_ && playerTypes_.at(sorryState_.getPlayerTurn()) != PlayerType::Human) {
    actionScoresChanged({});
    return;
  }
  const auto actions = sorryState_.getActions();
  std::vector<ActionScore> actionScores;
  actionScores.reserve(actions.size());
  for (const auto &action : actions) {
    actionScores.push_back({
      .action = action,
      .score = 0
    });
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
  // if (hiddenHand_ && playerTypes_.at(backendEnumToSorryEnum(playerColor)) != PlayerType::Human && playerTypes_.at(backendEnumToSorryEnum(playerColor)) != PlayerType::MctsAssistedHuman) {
  //   return { QString::fromStdString(toString(sorry::Card::kUnknown)),
  //            QString::fromStdString(toString(sorry::Card::kUnknown)),
  //            QString::fromStdString(toString(sorry::Card::kUnknown)),
  //            QString::fromStdString(toString(sorry::Card::kUnknown)),
  //            QString::fromStdString(toString(sorry::Card::kUnknown)) };
  // }
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

  const ActionScore &actionScore = actionScores_.at(index.row());
  if (role == NameRole) {
    const sorry::Action &action = actionScore.action;
    if (action.actionType == sorry::Action::ActionType::kDiscard) {
      return tr("Discard %1").arg(QString::fromStdString(sorry::toString(action.card)));
    }
    return QVariant(QString::fromStdString(action.toString()));
  } else if (role == ScoreRole) {
    return QVariant(actionScore.score);
  } else if (role == IsBestRole) {
    return QVariant(index.row() == bestIndex_);
  }
  return QVariant();
}

void ActionsList::setActionsAndScores(const std::vector<ActionScore> &actionsAndScores) {
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
  for (size_t givenActionIndex=0; givenActionIndex<actionsAndScores.size(); ++givenActionIndex) {
    const auto &givenActionScore = actionsAndScores.at(givenActionIndex);
    const sorry::Action &givenAction = givenActionScore.action;
    bool foundAction = false;
    for (size_t existingActionIndex=0; existingActionIndex<actionScores_.size(); ++existingActionIndex) {
      ActionScore &actionScore = actionScores_.at(existingActionIndex);
      if (actionScore.action == givenAction) {
        // Found our action, update the score
        actionScore.score = givenActionScore.score;
        if (givenActionIndex == bestIndex) {
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
    actionScores_.push_back(givenActionScore);
    if (givenActionIndex == bestIndex) {
      bestIndex_ = actionScores_.size()-1;
    }
    endInsertRows();
  }
  for (int i=actionSeen.size()-1; i>=0; --i) {
    if (!actionSeen[i]) {
      beginRemoveRows(QModelIndex(), i, i);
      actionScores_.erase(actionScores_.begin() + i);
      endRemoveRows();
    }
  }
}

std::optional<sorry::Action> ActionsList::getAction(int index) const {
  std::unique_lock lock(mutex_);
  if (index < 0 || index >= actionScores_.size()) {
    return {};
  }
  return actionScores_.at(index).action;
}
