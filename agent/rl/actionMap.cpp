#include "actionMap.hpp"

#include <sorry/engine/sorry.hpp>

#include <iostream>
#include <stdexcept>

namespace {

sorry::engine::Card cardIndexToCard(size_t index) {
  // Since the values of card enums are not a contiguous range, we can't just cast the index to a card.
  switch (index) {
    case 0: return sorry::engine::Card::kOne;
    case 1: return sorry::engine::Card::kTwo;
    case 2: return sorry::engine::Card::kThree;
    case 3: return sorry::engine::Card::kFour;
    case 4: return sorry::engine::Card::kFive;
    case 5: return sorry::engine::Card::kSeven;
    case 6: return sorry::engine::Card::kEight;
    case 7: return sorry::engine::Card::kTen;
    case 8: return sorry::engine::Card::kEleven;
    case 9: return sorry::engine::Card::kTwelve;
    case 10: return sorry::engine::Card::kSorry;
    default: throw std::runtime_error("Invalid card index");
  }
}

size_t cardToCardIndex(sorry::engine::Card card) {
  // Since the values of card enums are not a contiguous range, we can't just cast the card to an index.
  switch (card) {
    case sorry::engine::Card::kOne: return 0;
    case sorry::engine::Card::kTwo: return 1;
    case sorry::engine::Card::kThree: return 2;
    case sorry::engine::Card::kFour: return 3;
    case sorry::engine::Card::kFive: return 4;
    case sorry::engine::Card::kSeven: return 5;
    case sorry::engine::Card::kEight: return 6;
    case sorry::engine::Card::kTen: return 7;
    case sorry::engine::Card::kEleven: return 8;
    case sorry::engine::Card::kTwelve: return 9;
    case sorry::engine::Card::kSorry: return 10;
    default: throw std::runtime_error("Invalid card");
  }
}

}

ActionMap::ActionMap() {
  using sorry::engine::Action;

  auto addActions = [this](size_t count,
                           ActionMap::ActionRange::ActionGenerator actionGenerator,
                           ActionMap::ActionRange::IndexGenerator indexGenerator) {
    size_t startOfRange = actionRanges_.empty() ? 0 : actionRanges_.back().end;
    actionRanges_.emplace_back(startOfRange, startOfRange+count, actionGenerator, indexGenerator);
  };

  // Define actions for all values of `index`.
  // ========================================= DISCARD =========================================
  // The first 11 actions are for discarding specific card types.
  addActions(11,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      return Action::discard(playerColor, cardIndexToCard(index));
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kDiscard) {
        return std::nullopt;
      }
      return cardToCardIndex(action.card);
    });

  // ===================================== MOVE FROM START =====================================
  // The next action is for using a 1 card to move a piece out of Start.
  addActions(1,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      (void)index;
      constexpr int moveSource = 0;
      int firstPositionFromStart = sorry::engine::Sorry::getFirstPosition(playerColor);
      return Action::singleMove(playerColor, sorry::engine::Card::kOne, moveSource, firstPositionFromStart);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kOne ||
          action.move1Source != 0) {
        return std::nullopt;
      }
      // This assumes that action.move1Destination is the first position from start.
      return 0;
    }
  );

  // The next action is for using a 2 card to move a piece out of Start.
  addActions(1,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      (void)index;
      constexpr int moveSource = 0;
      int firstPositionFromStart = sorry::engine::Sorry::getFirstPosition(playerColor);
      return Action::singleMove(playerColor, sorry::engine::Card::kTwo, moveSource, firstPositionFromStart);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kTwo ||
          action.move1Source != 0) {
        return std::nullopt;
      }
      // This assumes that action.move1Destination is the first position from start.
      return 0;
    }
  );

  // ========================================== SORRY ==========================================
  // The next 60 actions are for using a Sorry card on a specific target public position.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      // Possible destinations are [1, 60]
      return Action::sorry(playerColor, index + 1);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSorry) {
        return std::nullopt;
      }
      // This assumes that action.card is Card::kSorry action.move1Source is 0.
      return action.move1Destination - 1;
    }
  );

  // ========================================== SWAP ===========================================
  // The next 3600(60*60) actions are for using an 11 card to swap a piece on a specific public position with an opponent piece on a different specific public position.
  addActions(60*60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      // Public positions are [1, 60]
      // index == sourcePosition * 60 + targetPosition
      int sourcePosition = index / 60 + 1;
      int targetPosition = index % 60 + 1;
      return Action::swap(playerColor, sourcePosition, targetPosition);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSwap) {
        return std::nullopt;
      }
      // This assumes that action.card is Card::kEleven.
      return (action.move1Source-1) * 60 + (action.move1Destination-1);
    }
  );

  // ====================================== SINGLE FOWARD ======================================
  // The next 65 actions are for using a 1 card to move forward 1, starting from a specific position in the range [1,65].
  addActions(65,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 1);
      return Action::singleMove(playerColor, sorry::engine::Card::kOne, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kOne ||
          action.move1Source == 0) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 64 actions are for using a 2 card to move forward 2, starting from a specific position in the range [1,64].
  addActions(64,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 2);
      return Action::singleMove(playerColor, sorry::engine::Card::kTwo, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kTwo ||
          action.move1Source == 0) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 63 actions are for using a 3 card to move forward 3, starting from a specific position in the range [1,63].
  addActions(63,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 3);
      return Action::singleMove(playerColor, sorry::engine::Card::kThree, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kThree) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 62 actions are for using a Sorry card to move forward 4, starting from a specific position in the range [1,62].
  addActions(62,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 4);
      return Action::singleMove(playerColor, sorry::engine::Card::kSorry, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kSorry) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 61 actions are for using a 5 card to move forward 5, starting from a specific position in the range [1,61].
  addActions(61,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 5);
      return Action::singleMove(playerColor, sorry::engine::Card::kFive, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kFive) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 60 actions are for using a 7 card to move forward 7, starting from a specific position in the range [1,60].
  // Note that while Green cannot move forward 7 from position 60 (because that would be beyond Green's home), any other player can move forward 7 from position 60.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 7);
      return Action::singleMove(playerColor, sorry::engine::Card::kSeven, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kSeven) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 60 actions are for using a 8 card to move forward 8, starting from a specific position in the range [1,60].
  // Note that while Green cannot move forward 8 from position 59 or 60 (because that would be beyond Green's home), any other player can move forward 8 from position 59 or 60.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 8);
      return Action::singleMove(playerColor, sorry::engine::Card::kEight, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kEight) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 60 actions are for using a 10 card to move forward 10, starting from a specific position in the range [1,60].
  // Note that while Green cannot move forward 10 from position 57,58,59, or 60 (because that would be beyond Green's home), any other player can move forward 10 from position 57,58,59, or 60.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 10);
      const auto a = Action::singleMove(playerColor, sorry::engine::Card::kTen, startingPosition, destination);
      return a;
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kTen ||
          sorry::engine::Sorry::posAfterMoveForPlayer(action.playerColor, action.move1Source, 10) != action.move1Destination) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 60 actions are for using a 11 card to move forward 11, starting from a specific position in the range [1,60].
  // Note that while Green cannot move forward 11 from position 56,57,58,59, or 60 (because that would be beyond Green's home), any other player can move forward 11 from position 56,57,58,59, or 60.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 11);
      return Action::singleMove(playerColor, sorry::engine::Card::kEleven, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kEleven) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 60 actions are for using a 12 card to move forward 12, starting from a specific position in the range [1,60].
  // Note that while Green cannot move forward 12 from position 55,56,57,58,59, or 60 (because that would be beyond Green's home), any other player can move forward 12 from position 55,56,57,58,59, or 60.
  addActions(60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, 12);
      return Action::singleMove(playerColor, sorry::engine::Card::kTwelve, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kTwelve) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // ===================================== SINGLE BACKWARD =====================================
  // The next 65 actions are for using a 10 card to move backward 1, starting from a specific position in the range [1,65].
  addActions(65,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, -1);
      return Action::singleMove(playerColor, sorry::engine::Card::kTen, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kTen ||
          sorry::engine::Sorry::posAfterMoveForPlayer(action.playerColor, action.move1Source, -1) != action.move1Destination) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // The next 65 actions are for using a 4 card to move backward 4, starting from a specific position in the range [1,65].
  addActions(65,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      int startingPosition = 1 + index;
      int destination = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, startingPosition, -4);
      return Action::singleMove(playerColor, sorry::engine::Card::kFour, startingPosition, destination);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kSingleMove ||
          action.card != sorry::engine::Card::kFour) {
        return std::nullopt;
      }
      return action.move1Source - 1;
    }
  );

  // =================================== SEVEN - DOUBLE MOVE ===================================
  // First 1, second 6
  // The next 3900(65*60) actions are for using a 7 card to move one piece forward 1, starting from a specific position in the range [1,65] and a second piece forward 6, starting from a specific position in the range [1,60].
  // Note: This range includes the possibility of specifying the same source twice (60 invalid actions) as well as the same destination twice (60-1 invalid actions; both ending on 66 is valid).
  addActions(65*60,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      // index == (piece1SourcePosition-1) * 60 + (piece2SourcePosition-1)
      int piece1SourcePosition = index / 60 + 1;
      int piece2SourcePosition = index % 60 + 1;
      int piece1DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece1SourcePosition, 1);
      int piece2DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece2SourcePosition, 6);
      return Action::doubleMove(playerColor, sorry::engine::Card::kSeven, piece1SourcePosition, piece1DestinationPosition, piece2SourcePosition, piece2DestinationPosition);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kDoubleMove ||
          sorry::engine::Sorry::posAfterMoveForPlayer(action.playerColor, action.move1Source, 1) != action.move1Destination) {
        return std::nullopt;
      }
      // This assumes that action.card is Card::kSeven.
      // This assumes that the second piece is moved forward by 6.
      return (action.move1Source-1) * 60 + (action.move2Source-1);
    }
  );

  // First 2, second 5
  // The next 3904(64*61) actions are for using a 7 card to move one piece forward 2, starting from a specific position in the range [1,64] and a second piece forward 5, starting from a specific position in the range [1,61].
  // Note: This range includes the possibility of specifying the same source twice (61 invalid actions) as well as the same destination twice (61-1 invalid actions; both ending on 66 is valid).
  addActions(64*61,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      // index == (piece1SourcePosition-1) * 61 + (piece2SourcePosition-1)
      int piece1SourcePosition = index / 61 + 1;
      int piece2SourcePosition = index % 61 + 1;
      int piece1DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece1SourcePosition, 2);
      int piece2DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece2SourcePosition, 5);
      return Action::doubleMove(playerColor, sorry::engine::Card::kSeven, piece1SourcePosition, piece1DestinationPosition, piece2SourcePosition, piece2DestinationPosition);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kDoubleMove ||
          sorry::engine::Sorry::posAfterMoveForPlayer(action.playerColor, action.move1Source, 2) != action.move1Destination) {
        return std::nullopt;
      }
      // This assumes that action.card is Card::kSeven.
      // This assumes that the second piece is moved forward by 5.
      return (action.move1Source-1) * 61 + (action.move2Source-1);
    }
  );

  // First 3, second 4
  // The next 3904(63*62) actions are for using a 7 card to move one piece forward 3, starting from a specific position in the range [1,63] and a second piece forward 4, starting from a specific position in the range [1,62].
  // Note: This range includes the possibility of specifying the same source twice (62 invalid actions) as well as the same destination twice (62-1 invalid actions; both ending on 66 is valid).
  addActions(63*62,
    [&](size_t index, sorry::engine::PlayerColor playerColor) {
      // index == (piece1SourcePosition-1) * 62 + (piece2SourcePosition-1)
      int piece1SourcePosition = index / 62 + 1;
      int piece2SourcePosition = index % 62 + 1;
      int piece1DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece1SourcePosition, 3);
      int piece2DestinationPosition = sorry::engine::Sorry::posAfterMoveForPlayer(playerColor, piece2SourcePosition, 4);
      return Action::doubleMove(playerColor, sorry::engine::Card::kSeven, piece1SourcePosition, piece1DestinationPosition, piece2SourcePosition, piece2DestinationPosition);
    },
    [&](const Action &action) -> std::optional<size_t> {
      if (action.actionType != Action::ActionType::kDoubleMove ||
          sorry::engine::Sorry::posAfterMoveForPlayer(action.playerColor, action.move1Source, 3) != action.move1Destination) {
        return std::nullopt;
      }
      // This assumes that action.card is Card::kSeven.
      // This assumes that the second piece is moved forward by 4.
      return (action.move1Source-1) * 62 + (action.move2Source-1);
    }
  );

  // ===========================================================================================

  constexpr bool kTestThatMappingIsABijection{true};
  if constexpr (kTestThatMappingIsABijection) {
    std::cout << "Running test on ActionMap" << std::endl;
    for (sorry::engine::PlayerColor playerColor : {sorry::engine::PlayerColor::kGreen,
                                                   sorry::engine::PlayerColor::kYellow, sorry::engine::PlayerColor::kRed, sorry::engine::PlayerColor::kBlue}) {
      for (int index=0; index<totalActionCount(); ++index) {
        sorry::engine::Action action = indexToActionForPlayer(index, playerColor);
        int newIndex = actionToIndex(action);
        if (index != newIndex) {
          throw std::runtime_error("Action index " + std::to_string(index) + " maps to action " + action.toString() + " which maps to index " + std::to_string(newIndex));
        }
      }
    }

    // // Manual test that an action->index->action round trip works.
    // const Action testAction = Action::singleMove(sorry::engine::PlayerColor::kBlue, sorry::engine::Card::kTen, 3, 2);
    // std::cout << "Getting index for action " << testAction.toString() << std::endl;
    // const int testIndex = actionToIndex(testAction);
    // const Action roundTripAction = indexToActionForPlayer(testIndex, sorry::engine::PlayerColor::kBlue);
    // std::cout << "testAction: " << testAction.toString() << std::endl;
    // std::cout << "testIndex: " << testIndex << std::endl;
    // std::cout << "roundTripAction: " << roundTripAction.toString() << std::endl;
    // const Action testAction2 = Action::singleMove(sorry::engine::PlayerColor::kBlue, sorry::engine::Card::kTwelve, 57, 9);
    // std::cout << "Getting index for testAction2: " << testAction2.toString() << std::endl;
    // const int testIndex2 = actionToIndex(testAction2);
    // std::cout << "testIndex2: " << testIndex2 << std::endl;
    // // Blue,SingleMove,Twelve,57,9
    // if (testIndex == testIndex2) {
    //   throw std::runtime_error("Manual test failed");
    // }
    std::cout << "ActionMap test passed" << std::endl;
  }
}

sorry::engine::Action ActionMap::indexToActionForPlayer(size_t index, sorry::engine::PlayerColor playerColor) const {
  for (const ActionRange &range : actionRanges_) {
    if (index >= range.start && index < range.end) {
      return range.actionGenerator(index - range.start, playerColor);
    }
  }
  throw std::runtime_error("Action index " + std::to_string(index) + " is out of range");
}

size_t ActionMap::actionToIndex(sorry::engine::Action action) const {
  std::optional<size_t> index;
  for (const ActionRange &range : actionRanges_) {
    index = range.indexGenerator(action);
    if (index.has_value()) {
      return range.start + index.value();
    }
  }
  throw std::runtime_error("Did not find an index for action " + action.toString());
}

int ActionMap::totalActionCount() const {
  if (actionRanges_.empty()) {
    return 0;
  }
  return actionRanges_.back().end;
}