#include "action.hpp"

#include <sstream>
#include <stdexcept>

namespace sorry::engine {

Action Action::discard(PlayerColor playerColor, Card card) {
  Action a;
  a.playerColor = playerColor;
  a.actionType = ActionType::kDiscard;
  a.card = card;
  return a;
}

Action Action::singleMove(PlayerColor playerColor, Card card, int moveSource, int moveDestination) {
  Action a;
  a.playerColor = playerColor;
  a.actionType = ActionType::kSingleMove;
  a.card = card;
  a.move1Source = moveSource;
  a.move1Destination = moveDestination;
  return a;
}

Action Action::doubleMove(PlayerColor playerColor, Card card, int move1Source, int move1Destination, int move2Source, int move2Destination) {
  Action a;
  a.playerColor = playerColor;
  a.actionType = ActionType::kDoubleMove;
  a.card = card;
  a.move1Source = move1Source;
  a.move1Destination = move1Destination;
  a.move2Source = move2Source;
  a.move2Destination = move2Destination;
  return a;
}

Action Action::sorry(PlayerColor playerColor, int moveDestination) {
  Action a;
  a.playerColor = playerColor;
  a.actionType = ActionType::kSorry;
  a.card = Card::kSorry;
  a.move1Destination = moveDestination;
  return a;
}

Action Action::swap(PlayerColor playerColor, int moveSource, int moveDestination) {
  Action a;
  a.playerColor = playerColor;
  a.actionType = ActionType::kSwap;
  a.card = Card::kEleven;
  a.move1Source = moveSource;
  a.move1Destination = moveDestination;
  return a;
}

std::string Action::toString() const {
  std::stringstream ss;
  ss << sorry::engine::toString(playerColor) << ',';
  if (actionType == ActionType::kDiscard) {
    ss << "Discard";
  } else if (actionType == ActionType::kSingleMove) {
    ss << "SingleMove";
  } else if (actionType == ActionType::kDoubleMove) {
    ss << "DoubleMove";
  } else if (actionType == ActionType::kSorry) {
    ss << "Sorry";
  } else if (actionType == ActionType::kSwap) {
    ss << "Swap";
  } else {
    throw std::runtime_error("Unknown action type");
  }
  if (actionType != ActionType::kSorry && actionType != ActionType::kSwap) {
    ss << ',' << sorry::engine::toString(card);
  }
  if (actionType == ActionType::kSingleMove || actionType == ActionType::kDoubleMove || actionType == ActionType::kSwap) {
    ss << ',' << move1Source << ',' << move1Destination;
    if (actionType == ActionType::kDoubleMove) {
      ss << ',' << move2Source << ',' << move2Destination;
    }
  } else if (actionType == ActionType::kSorry) {
    ss << ',' << move1Destination;
  }
  return ss.str();
}

bool operator==(const Action &lhs, const Action &rhs) {
  if (lhs.playerColor != rhs.playerColor) {
    return false;
  }
  if (lhs.actionType == Action::ActionType::kDiscard) {
    return rhs.actionType == Action::ActionType::kDiscard &&
           lhs.card == rhs.card;
  } else if (lhs.actionType == Action::ActionType::kSingleMove) {
    return rhs.actionType == Action::ActionType::kSingleMove &&
           lhs.card == rhs.card &&
           lhs.move1Source == rhs.move1Source &&
           lhs.move1Destination == rhs.move1Destination;
  } else if (lhs.actionType == Action::ActionType::kDoubleMove) {
    return rhs.actionType == Action::ActionType::kDoubleMove &&
           lhs.card == rhs.card &&
           lhs.move1Source == rhs.move1Source &&
           lhs.move1Destination == rhs.move1Destination &&
           lhs.move2Source == rhs.move2Source &&
           lhs.move2Destination == rhs.move2Destination;
  } else if (lhs.actionType == Action::ActionType::kSwap) {
    return rhs.actionType == Action::ActionType::kSwap &&
           lhs.move1Source == rhs.move1Source &&
           lhs.move1Destination == rhs.move1Destination;
  } else if (lhs.actionType == Action::ActionType::kSorry) {
    return rhs.actionType == Action::ActionType::kSorry &&
           lhs.move1Destination == rhs.move1Destination;
  }
  throw std::runtime_error("Missed a comparison");
}

} // namespace sorry::engine
