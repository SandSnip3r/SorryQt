#ifndef SORRY_ENGINE_DECK_HPP_
#define SORRY_ENGINE_DECK_HPP_

#include "card.hpp"

#include <array>
#include <random>

namespace sorry::engine {

class Deck {
public:
  Deck() { initialize(); }
  void initialize();
  void removeSpecificCard(Card card);
  Card drawRandomCard(std::mt19937 &eng);
  void discard(Card card);
  size_t size() const;
  bool empty() const;
  void shuffle();
  bool equalDiscarded(const Deck &other) const;
private:
  // Deck is made of three sections:
  // |--FaceDown--|--Out--|--Discarded--|
  // FaceDown cards are the ones which are not yet drawn. These are face-down and not visible to any player.
  // Out cards are the ones which are in some players' hand. Each player may see their own hand, but not the hand of another player.
  // Discarded cards are the ones which have been played. These are face-up and visible to all players.
  std::array<Card, 45> cards_;
  size_t firstOutIndex_;
  size_t firstDiscardIndex_;
  void removeCard(size_t index);
  void print() const;

  friend class Sorry;
  Card drawRandomCardAlsoFromOut(std::mt19937 &eng);
};

} // namespace sorry::engine

#endif // SORRY_ENGINE_DECK_HPP_