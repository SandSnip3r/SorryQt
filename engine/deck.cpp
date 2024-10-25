#include "deck.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace sorry::engine {

void Deck::initialize() {
  // 5 one's
  // 4 of every other card (2's, 3's, 4's, 5's, 7's, 8's, 10's, 11's, 12's,)
  // 4 Sorry cards
  cards_ = { Card::kOne,    Card::kOne,    Card::kOne,    Card::kOne,    Card::kOne,
             Card::kTwo,    Card::kTwo,    Card::kTwo,    Card::kTwo,
             Card::kThree,  Card::kThree,  Card::kThree,  Card::kThree,
             Card::kFour,   Card::kFour,   Card::kFour,   Card::kFour,
             Card::kFive,   Card::kFive,   Card::kFive,   Card::kFive,
             Card::kSeven,  Card::kSeven,  Card::kSeven,  Card::kSeven,
             Card::kEight,  Card::kEight,  Card::kEight,  Card::kEight,
             Card::kTen,    Card::kTen,    Card::kTen,    Card::kTen,
             Card::kEleven, Card::kEleven, Card::kEleven, Card::kEleven,
             Card::kTwelve, Card::kTwelve, Card::kTwelve, Card::kTwelve,
             Card::kSorry,  Card::kSorry,  Card::kSorry,  Card::kSorry };
  firstOutIndex_ = cards_.size();
  firstDiscardIndex_ = cards_.size();
}

void Deck::removeSpecificCard(Card card) {
  auto it = std::find(cards_.begin(), cards_.begin()+firstOutIndex_, card);
  if (it == cards_.end()) {
    throw std::runtime_error("Card not found in deck");
  }
  removeCard(std::distance(cards_.begin(), it));
}

Card Deck::drawRandomCard(std::mt19937 &eng) {
  std::uniform_int_distribution<size_t> dist(0, firstOutIndex_-1);
  const size_t drawnCardIndex = dist(eng);
  Card card = cards_[drawnCardIndex];
  removeCard(drawnCardIndex);
  return card;
}

void Deck::discard(Card card) {
  // Look for the card in the "out" range
  for (int i=firstDiscardIndex_-1; i>=static_cast<int>(firstOutIndex_); --i) {
    if (cards_.at(i) == card) {
      // Found our card, move it to the discard pile.
      std::swap(cards_.at(i), cards_.at(firstDiscardIndex_-1));
      --firstDiscardIndex_;

      // Bubble-up card so that discarded cards remain sorted
      size_t tmpIndex = firstDiscardIndex_;
      while (tmpIndex < cards_.size()-1 && cards_[tmpIndex] > cards_[tmpIndex+1]) {
        std::swap(cards_[tmpIndex], cards_[tmpIndex+1]);
        ++tmpIndex;
      }
      return;
    }
  }
  print();
  throw std::runtime_error("Cannot discard card which is not in \"out\" section");
}

size_t Deck::size() const {
  return firstOutIndex_;
}

bool Deck::empty() const {
  return firstOutIndex_ == 0;
}

void Deck::shuffle() {
  // Move everything from the discarded range to the end of the live range, shifting over the "out" range
  while (firstDiscardIndex_ < cards_.size()) {
    std::swap(cards_.at(firstOutIndex_), cards_.at(firstDiscardIndex_));
    ++firstOutIndex_;
    ++firstDiscardIndex_;
  }
}

void Deck::removeCard(size_t index) {
  std::swap(cards_.at(index), cards_.at(firstOutIndex_-1));
  --firstOutIndex_;
}

void Deck::print() const {
  for (auto c : cards_) {
    printf("%2d ", static_cast<int8_t>(c));
  }
  int i1 = (firstOutIndex_-1) * 3 + 2;
  int i2 = (firstDiscardIndex_-1) * 3 + 2;
  std::cout << '\n';
  if (i1 > 0) {
    std::cout << std::string(i1, ' ') << '^';
  }
  if (i2 > i1) {
    std::cout << std::string(i2-i1-1, ' ') << '^';
  }
  std::cout << std::endl;
}

Card Deck::drawRandomCardAlsoFromOut(std::mt19937 &eng) {
  // Note, this discards the card too.
  std::uniform_int_distribution<size_t> dist(0, firstDiscardIndex_-1);
  const size_t drawnCardIndex = dist(eng);
  Card card = cards_[drawnCardIndex];
  std::swap(cards_.at(drawnCardIndex), cards_.at(firstDiscardIndex_-1));
  --firstDiscardIndex_;
  if (firstOutIndex_ > firstDiscardIndex_) {
    --firstOutIndex_;
  }
  return card;
}

bool Deck::equalDiscarded(const Deck &other) const {
  if (firstDiscardIndex_ != other.firstDiscardIndex_) {
    return false;
  }
  for (size_t i=firstDiscardIndex_; i<cards_.size(); ++i) {
    if (cards_[i] != other.cards_[i]) {
      return false;
    }
  }
  return true;
}

} // namespace sorry::engine