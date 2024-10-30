#ifndef SORRY_ENGING_ACTION_HPP_
#define SORRY_ENGING_ACTION_HPP_

#include "card.hpp"
#include "playerColor.hpp"

#include <string>

namespace sorry::engine {

class Action {
public:
  enum class ActionType : uint8_t {
    kDiscard,
    kSingleMove,
    kDoubleMove,
    kSorry,
    kSwap
  };
  static Action discard(PlayerColor playerColor, Card card);
  static Action singleMove(PlayerColor playerColor, Card card, int moveSource, int moveDestination);
  static Action doubleMove(PlayerColor playerColor, Card card, int move1Source, int move1Destination,
                                                               int move2Source, int move2Destination);
  static Action sorry(PlayerColor playerColor, int moveDestination);
  static Action swap(PlayerColor playerColor, int moveSource, int moveDestination);
  std::string toString() const;

  PlayerColor playerColor;
  ActionType actionType;
  Card card;
  // Note: Movement destinations are before sliding.
  int move1Source;
  int move1Destination;
  int move2Source;
  int move2Destination;
};

bool operator==(const Action &lhs, const Action &rhs);
bool operator!=(const Action &lhs, const Action &rhs);

} // namespace sorry::engine


#endif // SORRY_ENGING_ACTION_HPP_