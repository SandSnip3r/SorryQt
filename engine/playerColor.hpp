#ifndef SORRY_ENGINE_PLAYER_COLOR_HPP_
#define SORRY_ENGINE_PLAYER_COLOR_HPP_

#include <cstdint>
#include <string_view>

namespace sorry::engine {

enum class PlayerColor : uint8_t {
  kGreen = 0,
  kRed = 1,
  kBlue = 2,
  kYellow = 3
};

std::string_view toString(PlayerColor playerColor);

} // namespace sorry::engine

#endif // SORRY_ENGINE_PLAYER_COLOR_HPP_