#include "common.hpp"

namespace sorry::engine::common {

int rotationCount(PlayerColor from, PlayerColor to) {
  int rotationCount = static_cast<int>(to) - static_cast<int>(from);
  if  (rotationCount < 0) {
    rotationCount += 4;
  }
  return rotationCount;
}

int rotatePosition(int position, int rotationCount) {
  if (position == 0 || position > 60) {
    // This is the player's private position. No need to rotate it.
    return position;
  }
  // Rotate the position.
  position -= 1;
  position += rotationCount * 15;
  position %= 60;
  position += 1;
  return position;
}

PlayerColor rotatePlayerColor(PlayerColor playerColor, int rotationCount) {
  int newPlayerColorInt = static_cast<int>(playerColor) + rotationCount;
  newPlayerColorInt %= 4;
  return static_cast<PlayerColor>(newPlayerColorInt);
}

} // namespace sorry::engine::common
