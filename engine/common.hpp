#ifndef SORRY_ENGINE_COMMON_HPP_
#define SORRY_ENGINE_COMMON_HPP_

#include "playerColor.hpp"

namespace sorry::engine::common {

// Returns the number of 90 degree clockwise rotations needed to go from 'from' to 'to'.
int rotationCount(PlayerColor from, PlayerColor to);

// Returns the new piece position after rotating the board 90 degrees 'rotationCount' times.
int rotatePosition(int position, int rotationCount);

// Returns the player color after rotating the board 90 degrees 'rotationCount' times.
PlayerColor rotatePlayerColor(PlayerColor playerColor, int rotationCount);

} // namespace sorry::engine::common

#endif // SORRY_ENGINE_COMMON_HPP_