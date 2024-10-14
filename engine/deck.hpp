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
private:
  std::array<Card, 45> cards_;
  size_t firstOutIndex_;
  size_t firstDiscardIndex_;
  void removeCard(size_t index);
  void print() const;

  friend bool operator==(const Deck &lhs, const Deck &rhs);
};

} // namespace sorry::engine

#endif // SORRY_ENGINE_DECK_HPP_