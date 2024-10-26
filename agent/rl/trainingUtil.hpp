#ifndef TRAINING_UTIL_HPP_
#define TRAINING_UTIL_HPP_

#include "trajectory.hpp"

#include <sorry/engine/action.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

class TrainingUtil {
public:
  TrainingUtil(pybind11::module &jaxModule);
  void setSeed(int seed);

  std::pair<pybind11::object, sorry::engine::Action> getGradientAndAction(
      const sorry::engine::Sorry &sorry,
      const std::vector<sorry::engine::Action> *validActions = nullptr);

  void train(const Trajectory &trajectory);
  void saveCheckpoint();
private:
  pybind11::object trainingUtilInstance_;
};

#endif // TRAINING_UTIL_HPP_