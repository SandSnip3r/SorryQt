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

SorryBackend::SorryBackend(QObject *parent) : QObject(parent), eng_(createRandomEngine()) {
  connect(this, &SorryBackend::actionScoresChanged, &actionsList_, &ActionsList::setActionsAndScores);

  sorryState_.drawRandomStartingCards(eng_);
  calculateScores();
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
    auto actionsAndScores = mcts_.getActionsWithScores();
    emit actionScoresChanged(actionsAndScores);
    // actionsList_.setActionsAndScores(actionsAndScores);
    // emit actionListModelChanged();
    // std::sort(actionsAndScores.begin(), actionsAndScores.end(), [](const auto &lhs, const auto &rhs){
    //   return rhs.second < lhs.second;
    // });

    // {
    //   std::unique_lock<std::mutex> lock(actionsMutex_);
    //   actions_.clear();
    //   for (const auto &actionAndScore : actionsAndScores) {
    //     actions_.push_back(new ActionForQml(actionAndScore.first, actionAndScore.second));
    //   }
    // }
    // emit actionsChanged();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void SorryBackend::calculateScores() {
  // Start running MCTS to figure out the value of each action.
  mctsTerminator_.setDone(false);
  mctsThread_ = std::thread([&](){ mcts_.run(sorryState_, &mctsTerminator_); });
  // Start another thread to periodically get data from mcts.
  actionProberThread_ = std::thread(&SorryBackend::probeActions, this);
}

void SorryBackend::doAction(const ActionForQml *action) {
  // Terminate prober
  runProber_ = false;
  actionProberThread_.join();
  // Terminate mcts
  mctsTerminator_.setDone(true);
  mctsThread_.join();
  // Do action
  std::cout << "Doing action " << action->getAction().toString() << std::endl;
  sorryState_.doAction(action->getAction(), eng_);
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

int SorryBackend::getMoveCount() const {
  return sorryState_.getTotalActionCount();
}

QVector<int> SorryBackend::getCardIndicesForAction(const ActionForQml *qmlAction) const {
  if (qmlAction == nullptr) {
    return {};
  }
  const auto &action = qmlAction->getAction();
  const auto card = action.card;
  const auto hand = sorryState_.getHand();
  QVector<int> result;
  for (int i=0; i<hand.size(); ++i) {
    if (hand[i] == card) {
      result.push_back(i);
    }
  }
  return result;
}

QVector<int> SorryBackend::getSrcAndDestPositionsForAction(const ActionForQml *qmlAction) const {
  const sorry::Action &action = qmlAction->getAction();
  if (action.actionType == sorry::Action::ActionType::kDiscard) {
    return {};
  }
  QVector<int> result;
  std::array<int, 4> piecePositions = sorryState_.getPiecePositions();
  result.push_back(action.piece1Index);
  result.push_back(piecePositions[action.piece1Index]);
  result.push_back(action.move1Destination);
  if (action.actionType == sorry::Action::ActionType::kDoubleMove) {
    result.push_back(action.piece2Index);
    result.push_back(piecePositions[action.piece2Index]);
    result.push_back(action.move2Destination);
  }
  return result;
}

// --------------------------------------------------------------------------------------------------------------

ActionsList::ActionsList() {}

int ActionsList::rowCount(const QModelIndex &parent) const {
  int val = actions_.size() + 20;
  if (parent.isValid()) {
    val = 0;
  }
  std::cout << "getting row count " << actions_.size() << ", returning " << val << std::endl;
  return val;
}

QVariant ActionsList::data(const QModelIndex &index, int role) const {
  std::cout << "data at index " << index.row() << " for role " << role << ". Row count: " << rowCount() << std::endl;
  if (!index.isValid()) {
    return QVariant();
  }
  if (index.row() < 0 || index.row() >= rowCount()) {
    return QVariant();
  }

  if (actions_.empty()) {
    return QVariant();
  }
  
  int rowIndex = index.row();
  if (rowIndex >= actions_.size()) {
    rowIndex = actions_.size()-1;
  }
  std::cout << "want action " << rowIndex << std::endl;
  const ActionForQml* action = actions_.at(rowIndex);
  if (role == NameRole) {
    return QVariant(action->name());
  } else if (role == ScoreRole) {
    return QVariant(action->score());
  }
  return QVariant();
}

void ActionsList::setActionsAndScores(const std::vector<std::pair<sorry::Action, double>> &actionsAndScores) {
  std::cout << "Given " << actionsAndScores.size() << " actions and scores" << std::endl;
  for (const auto &actionAndScore : actionsAndScores) {
    const sorry::Action &action = actionAndScore.first;
    bool foundAction = false;
    for (ActionForQml *actionForQml : actions_) {
      if (actionForQml->getAction() == action) {
        // Found our action, update the score
        std::cout << "Updating score for action " << action.toString() << " to " << actionAndScore.second << std::endl;
        actionForQml->updateScore(actionAndScore.second);
        foundAction = true;
        break;
      }
    }
    if (foundAction) {
      continue;
    }
    std::cout << "Did not find action " << action.toString() << ". Adding" << std::endl;
    beginInsertRows(QModelIndex(), actionsAndScores.size(), actionsAndScores.size());
    actions_.push_back(new ActionForQml(actionAndScore.first, actionAndScore.second));
    endInsertRows();
  }
}

ActionsList* SorryBackend::actionListModel() {
  return &actionsList_;
}
