#ifndef SORRY_ENGING_ACTION_HPP_
#define SORRY_ENGING_ACTION_HPP_

#include "card.hpp"
#include "playerColor.hpp"

#include <string>

namespace sorry::engine {

class Action {
public:
  enum class ActionType : uint8_t {
    kDiscard = 0,
    kSingleMove = 1,
    kDoubleMove = 2,
    kSorry = 3,
    kSwap = 4
  };
  Action() = default;
  Action(PlayerColor playerColor, ActionType actionType, Card card, int move1Source, int move1Destination, int move2Source, int move2Destination);
  static Action discard(PlayerColor playerColor, Card card);
  static Action singleMove(PlayerColor playerColor, Card card, int moveSource, int moveDestination);
  static Action doubleMove(PlayerColor playerColor, Card card, int move1Source, int move1Destination,
                                                               int move2Source, int move2Destination);
  static Action sorry(PlayerColor playerColor, int moveDestination);
  static Action swap(PlayerColor playerColor, int moveSource, int moveDestination);
  std::string toString() const;
  void rotateBoard(PlayerColor from, PlayerColor to);

  PlayerColor playerColor;
  ActionType actionType;
  Card card;
  // Note: Movement destinations are before sliding.
  int move1Source{0};
  int move1Destination{0};
  int move2Source{0};
  int move2Destination{0};
};

bool operator==(const Action &lhs, const Action &rhs);
bool operator!=(const Action &lhs, const Action &rhs);

} // namespace sorry::engine


#endif // SORRY_ENGING_ACTION_HPP_