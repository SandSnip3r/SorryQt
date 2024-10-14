#ifndef SORRY_ENGINE_CARD_HPP_
#define SORRY_ENGINE_CARD_HPP_

#include <cstdint>
#include <string>

namespace sorry::engine {

enum class Card : uint8_t {
  kOne = 1,
  kTwo = 2,
  kThree = 3,
  kFour = 0,
  kFive = 5,
  kSeven = 7,
  kEight = 8,
  kTen = 10,
  kEleven = 11,
  kTwelve = 12,
  kSorry = 4,
};

std::string toString(Card c);

} // namespace sorry::engine

#endif // SORRY_ENGINE_CARD_HPP_