#ifndef JAX_MODEL_HPP_
#define JAX_MODEL_HPP_

#include "jaxTrajectory.hpp"

#include <sorry/engine/action.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

class JaxModel {
public:
  JaxModel(pybind11::module &jaxModule);
  void setSeed(int seed);

  std::pair<pybind11::object, sorry::engine::Action> getGradientAndAction(
      const std::array<sorry::engine::Card, 5> &playerHand,
      const std::array<int, 4> &playerPiecePositions,
      const std::vector<sorry::engine::Action> *validActions = nullptr);

  void train(const JaxTrajectory &trajectory);
private:
  pybind11::object modelInstance_;

  pybind11::array_t<float> observationToNumpyArray(const std::array<sorry::engine::Card, 5> &playerHand,
                                            const std::array<int, 4> &playerPiecePositions) const;
};

#endif // JAX_MODEL_HPP_