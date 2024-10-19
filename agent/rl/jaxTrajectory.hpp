#ifndef JAX_TRAJECTORY_HPP_
#define JAX_TRAJECTORY_HPP_

#include <sorry/engine/card.hpp>

#include <array>

class JaxTrajectory {
public:
  void pushStep(const std::array<sorry::engine::Card, 5> &playerHand,
                const std::array<int, 4> &playerPiecePositions,
                double reward) {
    // TODO
  }
};

#endif // JAX_TRAJECTORY_HPP_