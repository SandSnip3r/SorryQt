#include "sorry_backend.h"

#include <iostream>
#include <algorithm>
#include <array>
#include <functional>

using namespace sorry;

std::mt19937 createRandomEngine() {
  std::random_device rd;
  std::array<int, std::mt19937::state_size> seed_data;
  std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  return std::mt19937(seq);
}

SorryBackend::SorryBackend(QObject *parent) : QObject(parent) {
  // randomSeed_ = -1701881796;
  randomSeed_ = std::random_device()();
  eng_ = std::mt19937(randomSeed_);
  connect(this, &SorryBackend::actionScoresChanged, &actionsList_, &ActionsList::setActionsAndScores);
  sorryState_.drawRandomStartingCards(eng_);
  calculateScores();
}

SorryBackend::~SorryBackend() {
  terminateThreads();
}

void SorryBackend::test() {
  std::cout << "test" << std::endl;
  static int i=0;
  m_dataList << "hey" + QString::number(i++);
  std::cout << m_dataList.size() << std::endl;
  emit dataListChanged();
}

QVector<ActionForQml*> SorryBackend::getActions() {
  std::unique_lock<std::mutex> lock(actionsMutex_);
  return actions_;
}

void SorryBackend::probeActions() {
  runProber_ = true;
  while (runProber_) {
    auto actionsAndScores = mcts_.getActionScores();
    emit actionScoresChanged(actionsAndScores);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void SorryBackend::calculateScores() {
  // Start running MCTS to figure out the value of each action.
  mctsTerminator_.setDone(false);
  mctsThread_ = std::thread([&](){ mcts_.run(sorryState_, &mctsTerminator_); });
  // Start another thread to periodically get data from mcts.
  actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
}

void SorryBackend::terminateThreads() {
  // Terminate prober
  runProber_ = false;
  actionProberThread_.join();
  // Terminate mcts
  mctsTerminator_.setDone(true);
  mctsThread_.join();
}

void SorryBackend::doAction(int index) {
  terminateThreads();
  // Do action
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to do unknown action (index " << index << ")" << std::endl;
    return;
  }
  std::cout << "Doing action " << action->toString() << std::endl;
  sorryState_.doAction(*action, eng_);
  emit moveCountChanged();
  emit actionsChanged();
  // Restart prober
  calculateScores();
}

QVector<int> SorryBackend::getPiecePositions() const {
  const auto positions = sorryState_.getPiecePositions();
  QVector<int> result;
  for (int i=0; i<4; ++i) {
    result.push_back(positions[i]);
  }
  return result;
}

QVector<QString> SorryBackend::getCardStrings() const {
  QVector<QString> result;
  const auto hand = sorryState_.getHand();
  for (const auto card : hand) {
    result.push_back(QString::fromStdString(toString(card)));
  }
  return result;
}

QVector<int> SorryBackend::getCardIndicesForAction(int index) const {
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to get card indices for unknown action (index " << index << ")" << std::endl;
    return {};
  }
  const auto card = action->card;
  const auto hand = sorryState_.getHand();
  QVector<int> result;
  for (int i=0; i<hand.size(); ++i) {
    if (hand[i] == card) {
      result.push_back(i);
    }
  }
  return result;
}

QVector<int> SorryBackend::getSrcAndDestPositionsForAction(int index) const {
  const auto action = actionsList_.getAction(index);
  if (!action) {
    std::cout << "Want to get src and dest pos for unknown action (index " << index << ")" << std::endl;
    return {};
  }
  if (action->actionType == sorry::Action::ActionType::kDiscard) {
    return {};
  }
  QVector<int> result;
  std::array<int, 4> piecePositions = sorryState_.getPiecePositions();
  result.push_back(action->piece1Index);
  result.push_back(piecePositions[action->piece1Index]);
  result.push_back(action->move1Destination);
  if (action->actionType == sorry::Action::ActionType::kDoubleMove) {
    result.push_back(action->piece2Index);
    result.push_back(piecePositions[action->piece2Index]);
    result.push_back(action->move2Destination);
  }
  return result;
}

ActionsList* SorryBackend::actionListModel() {
  return &actionsList_;
}

// --------------------------------------------------------------------------------------------------------------

ActionsList::ActionsList() {}

int ActionsList::rowCount(const QModelIndex &parent) const {
  if (parent.isValid()) {
    std::cout << "Parent is valid" << std::endl;
  }
  // std::cout << "Returning row count " << actions_.size() << std::endl;
  return actions_.size();
}

QVariant ActionsList::data(const QModelIndex &index, int role) const {
  // std::cout << "data() index: " << index.row() << ", role: " << role << std::endl;
  if (!index.isValid()) {
    return QVariant();
  }
  if (index.row() < 0 || index.row() >= rowCount()) {
    return QVariant();
  }

  if (actions_.empty()) {
    return QVariant();
  }
  
  const ActionScore &actionScore = actions_.at(index.row());
  if (role == NameRole) {
    const sorry::Action &action = actionScore.action;
    return QVariant(QString::fromStdString(action.toString()));
  } else if (role == ScoreRole) {
    return QVariant(actionScore.score);
  } else if (role == AverageMovesRole) {
    return QVariant(actionScore.averageMoveCount);
  }
  return QVariant();
}

void ActionsList::setActionsAndScores(const std::vector<ActionScore> &actionsAndScores) {
  std::vector<bool> actionSeen(actions_.size(), false);
  // beginResetModel();
  for (const auto &givenActionScore : actionsAndScores) {
    const sorry::Action &givenAction = givenActionScore.action;
    bool foundAction = false;
    for (int i=0; i<actions_.size(); ++i) {
      ActionScore &actionScore = actions_[i];
      if (actionScore.action == givenAction) {
        // Found our action, update the score
        actionScore.score = givenActionScore.score;
        actionScore.averageMoveCount = givenActionScore.averageMoveCount;
        emit dataChanged(this->index(i), this->index(i), {ScoreRole, AverageMovesRole});
        actionSeen[i] = true;
        foundAction = true;
        break;
      }
    }
    if (foundAction) {
      continue;
    }
    beginInsertRows(QModelIndex(), actions_.size(), actions_.size());
    actions_.push_back(givenActionScore);
    endInsertRows();
  }
  for (int i=actionSeen.size()-1; i>=0; --i) {
    if (!actionSeen[i]) {
      beginRemoveRows(QModelIndex(), i, i);
      actions_.remove(i);
      endRemoveRows();
    }
  }
  // endResetModel();
}

std::optional<sorry::Action> ActionsList::getAction(int index) const {
  if (index < 0 || index >= actions_.size()) {
    return {};
  }
  return actions_.at(index).action;
}