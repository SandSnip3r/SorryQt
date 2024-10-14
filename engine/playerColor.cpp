#include "playerColor.hpp"

namespace sorry::engine {

std::string_view toString(PlayerColor player) {
  if (player == PlayerColor::kGreen) {
    return "Green";
  }
  if (player == PlayerColor::kRed) {
    return "Red";
  }
  if (player == PlayerColor::kBlue) {
    return "Blue";
  }
  if (player == PlayerColor::kYellow) {
    return "Yellow";
  }
  return "UNKNOWN";
}

} // namespace sorry::engine