#include "sorry.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>

namespace sorry::engine {

bool Sorry::playerIsDone(const Sorry::Player &player) {
  for (auto pos : player.piecePositions) {
    if (pos != 66) {
      return false;
    }
  }
  return true;
}

Sorry::Sorry(const std::vector<PlayerColor> &playerColors) : Sorry(playerColors.data(), playerColors.size()) {}

Sorry::Sorry(std::initializer_list<PlayerColor> playerColors) : Sorry(std::data(playerColors), playerColors.size()) {}

Sorry::Sorry(const PlayerColor *playerColors, size_t playerCount) {
  if (playerCount > 4) {
    throw std::runtime_error("Too many players. Must be 4 or less.");
  }
  // Check for duplicates
  for (size_t i=0; i<playerCount; ++i) {
    for (size_t j=i+1; j<playerCount; ++j) {
      if (playerColors[i] == playerColors[j]) {
        throw std::runtime_error("Player " + std::string(sorry::engine::toString(playerColors[i])) + " appears multiple times in the player list");
      }
    }
  }

  // Initialize players
  players_.resize(playerCount);
  for (size_t i=0; i<playerCount; ++i) {
    auto &player = players_.at(i);
    player.playerColor = playerColors[i];
  }
}

void Sorry::reset(std::mt19937 &eng) {
  // Initialize player piece positions
  for (sorry::engine::Sorry::Player &player : players_) {
    if (SorryRules::instance().startWithOnePieceOutOfStart) {
      player.piecePositions[0] = getFirstPosition(player.playerColor);
    } else {
      player.piecePositions[0] = 0;
    }
    player.piecePositions[1] = 0;
    player.piecePositions[2] = 0;
    player.piecePositions[3] = 0;
  }

  // Set current player index
  currentPlayerIndex_ = 0;

  // Initialize deck
  deck_.initialize();

  // Draw starting cards for each player
  for (Player &player : players_) {
    for (size_t i=0; i<player.hand.size(); ++i) {
      player.hand.at(i) = deck_.drawRandomCard(eng);
      if (deck_.empty()) {
        throw std::runtime_error("Drew too many cards");
      }
    }
  }
  resetCalledAtLeastOnce_ = true;
}

// TODO: Possibly provide a version of `reset` which takes the starting cards as an argument.
// void Sorry::setStartingCards(PlayerColor playerColor, const std::array<Card,5> &cards) {
//   // Remove cards from deck and insert into hand.
//   Player &player = getPlayer(playerColor);
//   for (size_t i=0; i<cards.size(); ++i) {
//     player.hand[i] = cards[i];
//     deck_.removeSpecificCard(cards[i]);
//   }

//   static std::vector<bool> haveStartingHandForPlayer(players_.size(), false);
//   auto it = std::find_if(players_.begin(), players_.end(), [&](const auto &p) {
//     return p.playerColor == playerColor;
//   });
//   if (it == players_.end()) {
//     throw std::runtime_error("Dont have player");
//   }
//   int index = std::distance(players_.begin(), it);
//   haveStartingHandForPlayer[index] = true;
//   resetCalledAtLeastOnce_ = std::all_of(haveStartingHandForPlayer.begin(), haveStartingHandForPlayer.end(), [](bool b) { return b; });
// }

void Sorry::setStartingPositions(PlayerColor playerColor, const std::array<int, 4> &positions) {
  Player &player = getPlayer(playerColor);
  for (size_t i=0; i<positions.size(); ++i) {
    player.piecePositions[i] = positions[i];
  }
}

void Sorry::setTurn(PlayerColor playerColor) {
  const int originalCurrentPlayerIndex = currentPlayerIndex_;
  while (currentPlayer().playerColor != playerColor) {
    currentPlayerIndex_ = getNextPlayerIndex(currentPlayerIndex_);
    if (currentPlayerIndex_ == originalCurrentPlayerIndex) {
      throw std::runtime_error("Player not in game");
    }
  }
}

std::string Sorry::toString() const {
  if (!resetCalledAtLeastOnce_) {
    throw std::runtime_error("Called toString() without starting hands set");
  }
  std::stringstream ss;
  ss << '{';
  ss << "Deck:" << deck_.size();
  for (const auto &player : players_) {
    ss << ',' << player.toString();
  }
  ss << '}';
  return ss.str();
}

std::string Sorry::handToString() const {
  throw std::runtime_error("Not yet implemented");
  // if (!resetCalledAtLeastOnce_) {
  //   throw std::runtime_error("Called handToString() without starting hands set");
  // }
  // std::stringstream ss;
  // for (int i=0; i<5; ++i) {
  //   ss << sorry::engine::toString(hand_[i]);
  //   if (i != 4) {
  //     ss << ',';
  //   }
  // }
  // return ss.str();
}

std::array<Card,5> Sorry::getHandForPlayer(PlayerColor playerColor) const {
  return getPlayer(playerColor).hand;
}

std::array<int, 4> Sorry::getPiecePositionsForPlayer(PlayerColor playerColor) const {
  return getPlayer(playerColor).piecePositions;
}

std::vector<PlayerColor> Sorry::getPlayers() const {
  std::vector<PlayerColor> result;
  for (const auto &player : players_) {
    result.emplace_back(player.playerColor);
  }
  return result;
}

PlayerColor Sorry::getPlayerTurn() const {
  return players_.at(currentPlayerIndex_).playerColor;
}

std::vector<Action> Sorry::getActions() const {
  if (!resetCalledAtLeastOnce_) {
    throw std::runtime_error("Called getActions() without a starting hand set");
  }
  if (gameDone()) {
    return {};
  }
  std::vector<Action> result;
  result.reserve(20); // Save some time by doing one allocation large enough for most invocations.
  const auto &currentPlayerData = currentPlayer();
  const auto &currentPlayerHand = currentPlayerData.hand;
  for (size_t i=0; i<currentPlayerHand.size(); ++i) {
    bool alreadyHandledThisCard = false;
    for (int j=static_cast<int>(i)-1; j>=0; --j) {
      if (currentPlayerHand.at(j) == currentPlayerHand.at(i)) {
        // Already handled one of these cards.
        alreadyHandledThisCard = true;
      }
    }
    if (alreadyHandledThisCard) {
      continue;
    }
    addActionsForCard(currentPlayerData, currentPlayerHand.at(i), result);
  }

  if (result.empty()) {
    // If no options, create successor states for discarding.
    for (size_t i=0; i<currentPlayerHand.size(); ++i) {
      bool alreadyDiscarded=false;
      for (size_t j=0; j<i; ++j) {
        if (currentPlayerHand.at(i) == currentPlayerHand.at(j)) {
          // Already discarded one of these.
          alreadyDiscarded = true;
          break;
        }
      }
      if (!alreadyDiscarded) {
        result.push_back(Action::discard(currentPlayerData.playerColor, currentPlayerHand.at(i)));
      }
    }
  }
  return result;
}

std::vector<Sorry::Move> Sorry::getMovesForAction(const Action &action) const {
  if (action.actionType == Action::ActionType::kDiscard) {
    return {};
  }
  auto posOfPlayerPiece = [&](PlayerColor playerColor, int pieceIndex) {
    const auto &player = getPlayer(playerColor);
    return player.piecePositions.at(pieceIndex);
  };
  auto indexAndColorOfPieceAtPos = [&](int pos) {
    for (const auto &player : players_) {
      for (size_t i=0; i<player.piecePositions.size(); ++i) {
        const auto piecePos = player.piecePositions.at(i);
        if (pos == piecePos) {
          return std::make_pair(static_cast<int>(i), player.playerColor);
        }
      }
    }
    throw std::runtime_error("Cannot find piece at pos "+std::to_string(pos));
  };
  auto firstIndexInStart = [&](PlayerColor playerColor) -> int {
    const auto &player = getPlayer(playerColor);
    for (size_t i=0; i<player.piecePositions.size(); ++i) {
      if (player.piecePositions.at(i) == 0) {
        return i;
      }
    }
    throw std::runtime_error("No piece in start");
  };
  std::vector<Move> result;
  if (action.actionType == Action::ActionType::kSingleMove ||
      action.actionType == Action::ActionType::kDoubleMove) {
    result.push_back(Move{.playerColor = action.playerColor,
                          .pieceIndex = action.piece1Index,
                          .srcPosition = posOfPlayerPiece(action.playerColor, action.piece1Index),
                          .destPosition = action.move1Destination});
    if (action.actionType == Action::ActionType::kDoubleMove) {
      result.push_back(Move{.playerColor = action.playerColor,
                            .pieceIndex = action.piece2Index,
                            .srcPosition = posOfPlayerPiece(action.playerColor, action.piece2Index),
                            .destPosition = action.move2Destination});
    }
  } else if (action.actionType == Action::ActionType::kSwap) {
    const auto startPos = posOfPlayerPiece(action.playerColor, action.piece1Index);
    result.push_back(Move{.playerColor = action.playerColor,
                          .pieceIndex = action.piece1Index,
                          .srcPosition = startPos,
                          .destPosition = action.move1Destination});
    const auto [index, color] = indexAndColorOfPieceAtPos(action.move1Destination);
    result.push_back(Move{.playerColor = color,
                          .pieceIndex = index,
                          .srcPosition = action.move1Destination,
                          .destPosition = startPos});
  } else if (action.actionType == Action::ActionType::kSorry) {
    const auto indexInStart = firstIndexInStart(action.playerColor);
    result.push_back(Move{.playerColor = action.playerColor,
                          .pieceIndex = indexInStart,
                          .srcPosition = 0,
                          .destPosition = action.move1Destination});
    const auto [index, color] = indexAndColorOfPieceAtPos(action.move1Destination);
    result.push_back(Move{.playerColor = color,
                          .pieceIndex = index,
                          .srcPosition = action.move1Destination,
                          .destPosition = 0});
  }
  return result;
}

void Sorry::doAction(const Action &action, std::mt19937 &eng) {
  const auto prevState = *this;
  if (!resetCalledAtLeastOnce_) {
    throw std::runtime_error("Called doAction() without a starting hand set");
  }
  Player &player = getPlayer(action.playerColor);
  auto checkAndKillOpponents = [&](int startPos, int slideLength) {
    for (int i=0; i<slideLength; ++i) {
      int pos = posAfterMoveForPlayer(action.playerColor, startPos, i);
      if (pos == 0 || pos > 60) {
        // Cannot kill opponents in their start, safe zone, or home.
        continue;
      }
      for (Player &opponentPlayer : players_) {
        if (opponentPlayer.playerColor == action.playerColor) {
          // Do not check against self
          continue;
        }
        for (int &opponentPiecePos : opponentPlayer.piecePositions) {
          if (opponentPiecePos == pos) {
            // This opponent piece is killed; send it back to the player's start.
            opponentPiecePos = 0;
          }
        }
      }
    }
  };
  if (action.actionType == Action::ActionType::kSingleMove || action.actionType == Action::ActionType::kDoubleMove) {
    // Move one or two pieces
    // Check if any opponents die.
    checkAndKillOpponents(action.move1Destination, std::max(1,slideLengthAtPos(action.playerColor, action.move1Destination)));
    // Move our piece to final spot
    player.piecePositions.at(action.piece1Index) = posAfterSlide(action.playerColor, action.move1Destination);

    if (action.actionType == Action::ActionType::kDoubleMove) {
      // Check if any opponents die.
      checkAndKillOpponents(action.move2Destination, std::max(1,slideLengthAtPos(action.playerColor, action.move2Destination)));
      // Move our piece to final spot
      player.piecePositions.at(action.piece2Index) = posAfterSlide(action.playerColor, action.move2Destination);
    }
  } else if (action.actionType == Action::ActionType::kSorry) {
    // Find the first piece at pos 0.
    bool found{false};
    for (size_t i=0; i<player.piecePositions.size(); ++i) {
      if (player.piecePositions.at(i) == 0) {
        player.piecePositions.at(i) = action.move1Destination;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("Could not find a piece in start");
    }

    // Find the opponent & piece at the destination position
    found = false;
    for (Player &opponentPlayer : players_) {
      if (opponentPlayer.playerColor == action.playerColor) {
        continue;
      }
      for (size_t i=0; i<opponentPlayer.piecePositions.size(); ++i) {
        if (opponentPlayer.piecePositions.at(i) == action.move1Destination) {
          // Found our target.
          opponentPlayer.piecePositions.at(i) = 0;
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("Could not find target piece");
    }
  } else if (action.actionType == Action::ActionType::kSwap) {
    int &ourPos = player.piecePositions.at(action.piece1Index);
    // Find who's piece is on the destination position
    bool found = false;
    for (Player &opponentPlayer : players_) {
      if (opponentPlayer.playerColor == action.playerColor) {
        continue;
      }
      for (int &otherPlayerPos : opponentPlayer.piecePositions) {
        if (otherPlayerPos == action.move1Destination) {
          if (found) {
            throw std::runtime_error("Multiple pieces at the destination postion");
          }
          std::swap(ourPos, otherPlayerPos);
          found = true;
        }
      }
    }
  }

  // Do a quick sanity check to make that no two pieces are in the same spot, apart from start and home.
  constexpr bool kRunSanityCheck{false};
  if constexpr (kRunSanityCheck) {
    std::set<int> publicPositionsWithPiece;
    for (const Player &player : players_) {
      for (int pos : player.piecePositions) {

        if (pos > 0 && pos < 61) {
          if (publicPositionsWithPiece.find(pos) != publicPositionsWithPiece.end()) {
            std::cout << " -  Previous state: " << prevState.toString() << std::endl;
            std::cout << " -  Applied action: " << action.toString() << std::endl;
            std::cout << " -  Result state: " << toString() << std::endl;
            throw std::runtime_error("Multiple pieces on position "+std::to_string(pos));
          }
          publicPositionsWithPiece.insert(pos);
        }
      }
    }
  }

  // Draw/discard.
  Card newCard = deck_.drawRandomCard(eng);
  if (SorryRules::instance().shuffleAfterDiscard) {
    deck_.discard(action.card);
    if (deck_.empty()) {
      deck_.shuffle();
    }
  } else {
    if (deck_.empty()) {
      deck_.shuffle();
    }
    deck_.discard(action.card);
  }
  int oldCardIndex = player.indexOfCardInHand(action.card);
  player.hand.at(oldCardIndex) = newCard;

  // Advance the player turn.
  const bool anotherTurn = action.card == Card::kTwo &&
                           action.actionType == Action::ActionType::kSingleMove &&
                           SorryRules::instance().twoGetsAnotherTurn;
  if (!anotherTurn) {
    currentPlayerIndex_ = getNextPlayerIndex(currentPlayerIndex_);
  }
}

bool Sorry::gameDone() const {
  for (const auto &player : players_) {
    if (playerIsDone(player)) {
      return true;
    }
  }
  return false;
}

PlayerColor Sorry::getWinner() const {
  std::optional<PlayerColor> winner;
  for (const auto &player : players_) {
    if (playerIsDone(player)) {
      if (winner) {
        throw std::runtime_error("Multiple players are done");
      }
      winner = player.playerColor;
    }
  }

  if (!winner) {
    throw std::runtime_error("No player won");
  }
  return *winner;
}

int Sorry::getFirstPosition(PlayerColor playerColor) const {
  // Different color players come out of start to different positions.
  if (playerColor == PlayerColor::kGreen) {
    return 2;
  } else if (playerColor == PlayerColor::kRed) {
    return 17;
  } else if (playerColor == PlayerColor::kBlue) {
    return 32;
  } else if (playerColor == PlayerColor::kYellow) {
    return 47;
  } else {
    throw std::runtime_error("Invalid player");
  }
}

void Sorry::addActionsForCard(const Player &player, Card card, std::vector<Action> &actions) const {
  auto tryAddMoveToAllPositions = [this, &actions, &player](Card card, int moveAmount) {
    for (size_t pieceIndex=0; pieceIndex<player.piecePositions.size(); ++pieceIndex) {
      auto moveResult = getMoveResultingPos(player, pieceIndex, moveAmount);
      if (moveResult) {
        // Use card `card` and move piece `pieceIndex` from `piecePositions_[pieceIndex]` to `*moveResult`
        actions.push_back(Action::singleMove(player.playerColor, card, pieceIndex, *moveResult));
      }
    }
  };
  // Create the action of simply moving forward by the value of the card.
  if (!(card == Card::kSorry && !SorryRules::instance().sorryCanMoveForward4)) {
    int moveAmount;
    if (card == Card::kFour) {
      moveAmount = -4;
    } else {
      moveAmount = static_cast<int>(card);
    }

    // Note: This does not produce any duplicate actions because no two pieces can be in the same position (moving out of Start is not handled here).
    tryAddMoveToAllPositions(card, moveAmount);
  }

  // Move piece out of start, if possible.
  if (card == Card::kOne || card == Card::kTwo) {
    const auto firstPosition = getFirstPosition(player.playerColor);
    // Is any piece already on the start position?
    bool canMoveOutOfHome = true;
    for (auto position : player.piecePositions) {
      if (position == firstPosition) {
        canMoveOutOfHome = false;
        break;
      }
    }
    if (canMoveOutOfHome) {
      for (size_t pieceIndex=0; pieceIndex<player.piecePositions.size(); ++pieceIndex) {
        if (player.piecePositions[pieceIndex] == 0) {
          // This piece is in start.
          actions.push_back(Action::singleMove(player.playerColor, card, pieceIndex, firstPosition));
          // Note: Breaking after moving one item from start prevents duplicate actions. Any other piece in start results in the same action and pieces not in start don't apply here.
          break;
        }
      }
    }
  }

  if (card == Card::kTen) {
    // 10 can also go backward 1.
    tryAddMoveToAllPositions(card, -1);
  }
  if (card == Card::kSeven) {
    // 7 can have the 7 split across two pieces
    for (int move1=4; move1<7; ++move1) {
      int move2 = 7-move1;
      for (size_t piece1Index=0; piece1Index<player.piecePositions.size(); ++piece1Index) {
        for (size_t piece2Index=0; piece2Index<player.piecePositions.size(); ++piece2Index) {
          if (piece1Index == piece2Index) {
            continue;
          }
          auto doubleMoveResult = getDoubleMoveResultingPos(player, piece1Index, move1, piece2Index, move2);
          if (doubleMoveResult) {
            // Use card `card` and move piece `piece1Index` from `piecePositions_[piece1Index]` to `doubleMoveResult->first` and move piece `piece2Index` from `piecePositions_[piece2Index]` to `doubleMoveResult->second`.
            actions.push_back(Action::doubleMove(player.playerColor, card, piece1Index, doubleMoveResult->first, piece2Index, doubleMoveResult->second));
          }
        }
      }
    }
  }
  if (card == Card::kSorry) {
    for (auto pos : player.piecePositions) {
      if (pos != 0) {
        continue;
      }
      for (const auto &otherPlayer : players_) {
        if (otherPlayer.playerColor == player.playerColor) {
          continue;
        }
        for (const auto otherPos : otherPlayer.piecePositions) {
          if (!(otherPos > 0 && otherPos < 61)) {
            continue;
          }
          // Can "Sorry" this piece
          actions.push_back(Action::sorry(player.playerColor, otherPos));
        }
      }
      break;
    }
  }
  if (card == Card::kEleven) {
    for (size_t i=0; i<player.piecePositions.size(); ++i) {
      const auto pos = player.piecePositions.at(i);
      if (!(pos > 0 && pos < 61)) {
        // Cannot swap using our pieces in start, safe zone, nor home.
        continue;
      }
      for (const auto &otherPlayer : players_) {
        if (otherPlayer.playerColor == player.playerColor) {
          continue;
        }
        for (const auto otherPos : otherPlayer.piecePositions) {
          if (!(otherPos > 0 && otherPos < 61)) {
            continue;
          }
          // Can swap places with this player's piece
          actions.push_back(Action::swap(player.playerColor, i, otherPos));
        }
      }
    }
  }
}

int Sorry::posAfterMoveForPlayer(PlayerColor playerColor, int startingPosition, int moveDistance) const {
  int newPosition = startingPosition + moveDistance;
  const int lastPublicPos = [&]() {
    //  Green goes from 60 to 61
    //    Red goes from 15 to 61
    //   Blue goes from 30 to 61
    // Yellow goes from 45 to 61
    if (playerColor == PlayerColor::kGreen) {
      return 60;
    } else if (playerColor == PlayerColor::kRed) {
      return 15;
    } else if (playerColor == PlayerColor::kBlue) {
      return 30;
    } else if (playerColor == PlayerColor::kYellow) {
      return 45;
    } else {
      throw std::runtime_error("Invalid player");
    }
  }();
  bool inSafeZone{false};
  if (startingPosition <= lastPublicPos && newPosition > lastPublicPos) {
    // Moving forward into the safe zone
    newPosition = newPosition + (60-lastPublicPos);
    inSafeZone = true;
  }
  if (startingPosition >= 61 && newPosition >= 61) {
    inSafeZone = true;
  }
  if (startingPosition >= 61 && newPosition < 61) {
    // Moving backward out of the safe zone
    newPosition = newPosition - (60-lastPublicPos);
  }
  if (newPosition < 1) {
    // Wrap around.
    // ex.  0 becomes 60
    // ex. -1 becomes 59
    newPosition += 60;
  }
  if (!inSafeZone && newPosition > 60) {
    newPosition -= 60;
  }
  return newPosition;
}

std::optional<int> Sorry::getMoveResultingPos(const Player &player, int pieceIndex, int moveDistance) const {
  const int startingPosition = player.piecePositions[pieceIndex];
  // 0 is start
  // Public positions are 1-60
  // 5 safe positions (61,62,63,64,65)
  // 66 is homes
  if (startingPosition == 0) {
    // Is in start. Can't move.
    return {};
  }
  if (startingPosition == 66) {
    // Is in home. Can't move.
    return {};
  }

  int newPos = posAfterMoveForPlayer(player.playerColor, startingPosition, moveDistance);
  if (newPos > 66) {
    // Cannot go beyond home.
    return {};
  }

  if (newPos == startingPosition) {
    throw std::runtime_error("New position == starting position");
  }
  if (newPos == 66) {
    return newPos;
  }

  // Do we land on one of our own pieces?
  for (int i=0; i<4; ++i) {
    if (newPos == player.piecePositions[i]) {
      // Cannot move here.
      return {};
    }
  }
  int slideLength = slideLengthAtPos(player.playerColor, newPos);
  for (int slidePos=1; slidePos<slideLength; ++slidePos) {
    const int alongSlidePos = posAfterMoveForPlayer(player.playerColor, newPos, slidePos);
    for (size_t otherPieceIndex=0; otherPieceIndex<player.piecePositions.size(); ++otherPieceIndex) {
      if (static_cast<int>(otherPieceIndex) == pieceIndex) {
        // Do not check collision with same piece.
        continue;
      }
      if (player.piecePositions[otherPieceIndex] == alongSlidePos) {
        // One of our pieces is on this slide. Cannot move here.
        return {};
      }
    }
  }
  return newPos;
}

std::optional<std::pair<int,int>> Sorry::getDoubleMoveResultingPos(const Player &player, int piece1Index, int move1Distance, int piece2Index, int move2Distance) const {
  const int startingPosition1 = player.piecePositions[piece1Index];
  const int startingPosition2 = player.piecePositions[piece2Index];
  if (startingPosition1 == 0 || startingPosition2 == 0) {
    // Is in start. Can't move.
    return {};
  }
  if (startingPosition1 == 66 || startingPosition2 == 66) {
    // Is in home. Can't move.
    return {};
  }
  int newPos1 = posAfterMoveForPlayer(player.playerColor, startingPosition1, move1Distance);
  int newPos2 = posAfterMoveForPlayer(player.playerColor, startingPosition2, move2Distance);
  // Valid positions are 0-66. 0 is start, 66 is home.
  if (newPos1 > 66 || newPos2 > 66) {
    // Cannot go beyond home.
    return {};
  }
  // 5 safe positions (65,64,63,62,61)
  if (newPos1 == newPos2 && newPos1 != 66) {
    // Cannot move both pieces to the same place.
    return {};
  }

  // Do we land on one of our own pieces?
  for (int i=0; i<4; ++i) {
    if (i == piece1Index || i == piece2Index) {
      // Do not check collision with moving pieces.
      continue;
    }
    if ((newPos1 != 66 && player.piecePositions[i] == newPos1) ||
        (newPos2 != 66 && player.piecePositions[i] == newPos2)) {
      // Cannot move here.
      return {};
    }
  }

  // Check if either of these pieces slide over one of our other non-moving pieces.
  int slideLength1 = slideLengthAtPos(player.playerColor, newPos1);
  for (int slidePos=0; slidePos<slideLength1; ++slidePos) {
    for (size_t otherPieceIndex=0; otherPieceIndex<player.piecePositions.size(); ++otherPieceIndex) {
      if (static_cast<int>(otherPieceIndex) == piece1Index || static_cast<int>(otherPieceIndex) == piece2Index) {
        // Do not check collision with self.
        continue;
      }
      if (player.piecePositions[otherPieceIndex] == posAfterMoveForPlayer(player.playerColor, newPos1, slidePos)) {
        // One of our pieces is on this slide. Cannot move here.
        return {};
      }
    }
  }
  int slideLength2 = slideLengthAtPos(player.playerColor, newPos2);
  for (int slidePos=0; slidePos<slideLength2; ++slidePos) {
    for (size_t otherPieceIndex=0; otherPieceIndex<player.piecePositions.size(); ++otherPieceIndex) {
      if (static_cast<int>(otherPieceIndex) == piece1Index || static_cast<int>(otherPieceIndex) == piece2Index) {
        // Do not check collision with self.
        continue;
      }
      if (player.piecePositions[otherPieceIndex] == posAfterMoveForPlayer(player.playerColor, newPos2, slidePos)) {
        // One of our pieces is on this slide. Cannot move here.
        return {};
      }
    }
  }

  // Check if one piece is on a slide and if the other one is going to slide on it.
  if (slideLength1 > 0) {
    // Piece 1 is going to slide, check if piece 2 is currently on the slide.
    bool inTheWayBefore{false};
    bool inTheWayAfter{false};
    for (int slidePos=0; slidePos<slideLength1; ++slidePos) {
      if (startingPosition2 == newPos1 + slidePos) {
        // Piece 2 is in our way before it moves.
        inTheWayBefore = true;
      }
      if (newPos2 == newPos1 + slidePos) {
        // Piece 2 is in our way after it moves.
        inTheWayAfter = true;
      }
    }
    if (inTheWayBefore && inTheWayAfter) {
      return {};
    }
  }

  if (slideLength2 > 0) {
    // Piece 2 is going to slide, check if piece 1 is currently on the slide.
    bool inTheWayBefore{false};
    bool inTheWayAfter{false};
    for (int slidePos=0; slidePos<slideLength2; ++slidePos) {
      if (startingPosition1 == newPos2 + slidePos) {
        // Piece 1 is in our way before it moves.
        inTheWayBefore = true;
      }
      if (newPos1 == newPos2 + slidePos) {
        // Piece 1 is in our way after it moves.
        inTheWayAfter = true;
      }
    }
    if (inTheWayBefore && inTheWayAfter) {
      return {};
    }
  }

  const int afterSlide1 = posAfterSlide(player.playerColor, newPos1);
  const int afterSlide2 = posAfterSlide(player.playerColor, newPos2);
  if (afterSlide1 == afterSlide2 && afterSlide1 != 66) {
    // Both end at the same spot. Not acceptable.
    return {};
  }
  return std::make_pair(newPos1, newPos2);
}

int Sorry::slideLengthAtPos(PlayerColor playerColor, int pos) const {
  // Do we land on a slide?
  //  Green: Start @ 59, length 4
  //  Green: Start @  7, length 5
  //    Red: Start @ 14, length 4
  //    Red: Start @ 22, length 5
  //   Blue: Start @ 29, length 4
  //   Blue: Start @ 37, length 5
  // Yellow: Start @ 44, length 4
  // Yellow: Start @ 52, length 5
  if (playerColor != PlayerColor::kGreen) {
    if (pos == 59) return 4;
    if (pos ==  7) return 5;
  }
  if (playerColor != PlayerColor::kRed) {
    if (pos == 14) return 4;
    if (pos == 22) return 5;
  }
  if (playerColor != PlayerColor::kBlue) {
    if (pos == 29) return 4;
    if (pos == 37) return 5;
  }
  if (playerColor != PlayerColor::kYellow) {
    if (pos == 44) return 4;
    if (pos == 52) return 5;
  }
  // No slide
  return 0;
}

int Sorry::posAfterSlide(PlayerColor playerColor, int pos) const {
  int slideLength = slideLengthAtPos(playerColor, pos);
  if (slideLength > 0) {
    return posAfterMoveForPlayer(playerColor, pos, slideLength-1);
  }
  return pos;
}

int Sorry::getNextPlayerIndex(int currentIndex) const {
  ++currentIndex;
  if (static_cast<size_t>(currentIndex) >= players_.size()) {
    currentIndex = 0;
  }
  return currentIndex;
}

Sorry::Player& Sorry::currentPlayer() {
  return players_.at(currentPlayerIndex_);
}

const Sorry::Player& Sorry::currentPlayer() const {
  return players_.at(currentPlayerIndex_);
}

Sorry::Player& Sorry::getPlayer(PlayerColor playerColor) {
  for (auto &player : players_) {
    if (player.playerColor == playerColor) {
      return player;
    }
  }
  throw std::runtime_error("Trying to get player data for an invalid player "+std::string(sorry::engine::toString(playerColor)));
}

const Sorry::Player& Sorry::getPlayer(PlayerColor playerColor) const {
  for (const auto &player : players_) {
    if (player.playerColor == playerColor) {
      return player;
    }
  }
  throw std::runtime_error("Trying to get player data for an invalid player "+std::string(sorry::engine::toString(playerColor)));
}

bool operator==(const sorry::engine::Sorry &lhs, const sorry::engine::Sorry &rhs) {
  if (lhs.players_.size() != rhs.players_.size()) {
    return false;
  }
  if (!(lhs.deck_ == rhs.deck_)) {
    return false;
  }
  for (size_t playerIndex=0; playerIndex<lhs.players_.size(); ++playerIndex) {
    const sorry::engine::Sorry::Player &lhsPlayer = lhs.players_.at(playerIndex);
    const sorry::engine::Sorry::Player &rhsPlayer = rhs.players_.at(playerIndex);
    for (size_t i=0; i<lhsPlayer.hand.size(); ++i) {
      if (lhsPlayer.hand[i] != rhsPlayer.hand[i]) {
        return false;
      }
    }
    for (size_t i=0; i<lhsPlayer.piecePositions.size(); ++i) {
      if (lhsPlayer.piecePositions[i] != rhsPlayer.piecePositions[i]) {
        return false;
      }
    }
  }
  return true;
}


size_t Sorry::Player::indexOfCardInHand(Card card) const {
  for (size_t i=0; i<hand.size(); ++i) {
    if (hand.at(i) == card) {
      return i;
    }
  }
  throw std::runtime_error("Card "+sorry::engine::toString(card)+" not in hand");
}

std::string Sorry::Player::toString() const {
  std::stringstream ss;
  ss << '(' << sorry::engine::toString(playerColor) << ':';
  for (size_t i=0; i<hand.size(); ++i) {
    ss << sorry::engine::toString(hand[i]);
    if (i != hand.size()-1) {
      ss << ',';
    }
  }
  ss << '|';
  for (size_t i=0; i<piecePositions.size(); ++i) {
    ss << piecePositions[i];
    if (i != piecePositions.size()-1) {
      ss << ',';
    }
  }
  ss << ')';
  return ss.str();
}

} // namespace sorry::engine
