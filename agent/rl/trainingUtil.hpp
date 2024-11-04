#ifndef PYTHON_WRAPPER_TRAINING_UTIL_HPP_
#define PYTHON_WRAPPER_TRAINING_UTIL_HPP_

#include "trajectory.hpp"

#include <sorry/engine/action.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace python_wrapper {

class TrainingUtil {
public:
  TrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter, std::optional<std::string> checkpointDirName = std::nullopt);
  void setSeed(int seed);

  std::pair<pybind11::object, sorry::engine::Action> getPolicyGradientAndAction(
      pybind11::object observation,
      sorry::engine::PlayerColor playerColor,
      int episodeIndex,
      const std::vector<sorry::engine::Action> *validActions = nullptr);

  std::pair<pybind11::object, float> getValueGradientAndValue(pybind11::object observation);

  void train(const Trajectory &trajectory, int episodeIndex);
  void trainOld(const Trajectory &trajectory, int episodeIndex);
  void saveCheckpoint();
private:
  static constexpr float kGamma{0.99};
  static constexpr float kPolicyNetworkLearningRate{0.001};
  static constexpr float kValueNetworkLearningRate{0.00005};
  pybind11::module jaxModule_;
  pybind11::object summaryWriter_;
  pybind11::object trainingUtilInstance_;
  pybind11::object policyOptimizer_;
  pybind11::object valueOptimizer_;
};

} // namespace python_wrapper

#endif // PYTHON_WRAPPER_TRAINING_UTIL_HPP_