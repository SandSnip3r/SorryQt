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

  sorry::engine::Action getAction(const std::array<sorry::engine::Card, 5> &playerHand, const std::array<int, 4> &playerPiecePositions);

  void train(const JaxTrajectory &trajectory);
private:
  pybind11::object modelInstance_;

  pybind11::array_t<float> observationToNumpyArray(const std::array<sorry::engine::Card, 5> &playerHand,
                                            const std::array<int, 4> &playerPiecePositions) const;

  sorry::engine::Action numpyArrayToAction(const pybind11::array_t<float> &numpyActionVector) const;
};

#endif // JAX_MODEL_HPP_