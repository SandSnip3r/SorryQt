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
  void reset(std::mt19937 &eng);
  // void setStartingCards(PlayerColor playerColor, const std::array<Card,5> &cards);
  void setStartingPositions(PlayerColor playerColor, const std::array<int, 4> &positions);
  void setTurn(PlayerColor playerColor);

  std::string toString() const;
  std::string handToString() const;

  std::vector<PlayerColor> getPlayers() const;
  PlayerColor getPlayerTurn() const;
  std::array<Card,5> getHandForPlayer(PlayerColor playerColor) const;

  /* Internally, piece positions are tracked as follows:
  *   - 0 is the player's start; this is the case for all players.
  *     That is to say, Yellow's start is 0, Green's start is 0, etc.
  *   - The public/shared positions on the board are 1-60. We've chosen
  *     2 as the position that Green moves to after coming out of start.
  *     The numbers increment in a clockwise direction. The position
  *     counterclockwise of 2 is 1, we call this "Green's Gooch".
  *     For players other than Green, the position immediately
  *     after 60 is 1; Green would instead move up into his safe zone.
  *   - There are 5 safe positions (61,62,63,64,65). Similar to
  *     the start position, these are private positions that each player
  *     has their own instance of.
  *   - 66 is the player's start. Similar to the start & safe positions,
  *     this is a private position that each player has their own instance of.
  */
  std::array<int, 4> getPiecePositionsForPlayer(PlayerColor playerColor) const;

  // Internally, for any DoubleMove, move1 is always be the shorter move.
  std::vector<Action> getActions() const;
  int getFaceDownCardsCount() const;

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

  static int getFirstPosition(PlayerColor playerColor);
  static int posAfterMoveForPlayer(PlayerColor playerColor, int startingPosition, int moveDistance);
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
  bool resetCalledAtLeastOnce_{false};
  int currentPlayerIndex_;
  Deck deck_;
  void addActionsForCard(const Player &player, Card card, std::vector<Action> &actions) const;
  int getIndexOfPieceAtPosition(PlayerColor playerColor, int position) const;
  std::optional<int> getMoveResultingPos(const Player &player, int pieceIndex, int moveDistance) const;
  std::optional<std::pair<int,int>> getDoubleMoveResultingPos(const Player &player, int piece1Index, int move1Distance, int piece2Index, int move2Distance) const;
  int slideLengthAtPos(PlayerColor playerColor, int pos) const;
  int posAfterSlide(PlayerColor playerColor, int pos) const;
  int getNextPlayerIndex(int currentIndex) const;
  Player& currentPlayer();
  const Player& currentPlayer() const;
  Player& getPlayer(PlayerColor player);
  const Player& getPlayer(PlayerColor player) const;
  static bool playerIsDone(const Player &player);

  friend bool operator==(const Sorry &lhs, const Sorry &rhs);
};

} // namespace sorry::engine

#endif // SORRY_ENGINE_SORRY_HPP_