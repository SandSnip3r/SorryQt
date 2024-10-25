#include "card.hpp"

#include <stdexcept>

namespace sorry::engine {

std::string toString(Card c) {
  if (c == Card::kOne) {
    return "One";
  }
  if (c == Card::kTwo) {
    return "Two";
  }
  if (c == Card::kThree) {
    return "Three";
  }
  if (c == Card::kFour) {
    return "Four";
  }
  if (c == Card::kFive) {
    return "Five";
  }
  if (c == Card::kSeven) {
    return "Seven";
  }
  if (c == Card::kEight) {
    return "Eight";
  }
  if (c == Card::kTen) {
    return "Ten";
  }
  if (c == Card::kEleven) {
    return "Eleven";
  }
  if (c == Card::kTwelve) {
    return "Twelve";
  }
  if (c == Card::kSorry) {
    return "Sorry";
  }
  if (c == Card::kUnknown) {
    return "?";
  }
  throw std::runtime_error("toString(Card): Invalid card "+std::to_string(static_cast<int>(c)));
}

} // namespace sorry::engine