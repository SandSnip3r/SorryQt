#ifndef SORRY_ENGINE_SORRY_HPP_
#define SORRY_ENGINE_SORRY_HPP_

#include "action.hpp"
#include "card.hpp"
#include "deck.hpp"

#include <array>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sorry::engine {

// Singleton class for the rules of the game.
struct SorryRules {
  bool sorryCanMoveForward4{true};
  bool twoGetsAnotherTurn{true};
  bool startWithOnePieceOutOfStart{true};

  // The game is slightly different depending on the order of discard & shuffle. If false, shuffle first then discard.
  bool shuffleAfterDiscard{true};
  static SorryRules& instance() {
    static SorryRules rules;
    return rules;
  }
private:
  SorryRules() {}
};

class Sorry {
public:
  Sorry(const std::vector<PlayerColor> &playerColors);
  Sorry(std::initializer_list<PlayerColor> playerColors);
  void drawRandomStartingCards(std::mt19937 &eng);
  void setStartingCards(PlayerColor playerColor, const std::array<Card,5> &cards);
  void setStartingPositions(PlayerColor playerColor, const std::array<int, 4> &positions);
  void setTurn(PlayerColor playerColor);

  std::string toString() const;
  std::string handToString() const;

  std::vector<PlayerColor> getPlayers() const;
  PlayerColor getPlayerTurn() const;
  std::array<Card,5> getHandForPlayer(PlayerColor playerColor) const;
  std::array<int, 4> getPiecePositionsForPlayer(PlayerColor playerColor) const;
  std::vector<Action> getActions() const;

  struct Move {
    PlayerColor playerColor;
    int pieceIndex;
    int srcPosition;
    int destPosition;
  };
  std::vector<Move> getMovesForAction(const Action &action) const;

  void doAction(const Action &action, std::mt19937 &eng);

  bool gameDone() const;
  PlayerColor getWinner() const;

private:
  Sorry(const PlayerColor *playerColors, size_t playerCount);
  struct Player {
    PlayerColor playerColor;
    std::array<Card,5> hand;
    std::array<int, 4> piecePositions;
    size_t indexOfCardInHand(Card card) const;
    std::string toString() const;
  };
  std::vector<Player> players_;
  bool haveStartingHands_{false};
  int currentPlayerIndex_;
  Deck deck_;
  void addActionsForCard(const Player &player, Card card, std::vector<Action> &actions) const;
  std::optional<int> getMoveResultingPos(const Player &player, int pieceIndex, int moveDistance) const;
  std::optional<std::pair<int,int>> getDoubleMoveResultingPos(const Player &player, int piece1Index, int move1Distance, int piece2Index, int move2Distance) const;
  int slideLengthAtPos(PlayerColor playerColor, int pos) const;
  int posAfterSlide(PlayerColor playerColor, int pos) const;
  int getNextPlayerIndex(int currentIndex) const;
  Player& currentPlayer();
  const Player& currentPlayer() const;
  Player& getPlayer(PlayerColor player);
  const Player& getPlayer(PlayerColor player) const;
  int posAfterMoveForPlayer(PlayerColor playerColor, int startingPosition, int moveDistance) const;
  int getFirstPosition(PlayerColor playerColor) const;
  static bool playerIsDone(const Player &player);

  friend bool operator==(const Sorry &lhs, const Sorry &rhs);
};

} // namespace sorry::engine

#endif // SORRY_ENGINE_SORRY_HPP_