#ifndef PYTHON_WRAPPER_REINFORCE_WITH_BASELINE_TRAINING_UTIL_HPP_
#define PYTHON_WRAPPER_REINFORCE_WITH_BASELINE_TRAINING_UTIL_HPP_

#include "trajectory.hpp"

#include <sorry/engine/action.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>

namespace python_wrapper {

class ReinforceWithBaselineTrainingUtil {
public:
  ReinforceWithBaselineTrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter, std::optional<std::string> checkpointDirName = std::nullopt);
  void setSeed(int seed);
  pybind11::object getPythonTrainingUtilInstance() const { return trainingUtilInstance_; }

  std::pair<sorry::engine::Action, pybind11::object> getActionAndKeyUsed(
      const std::vector<int> &observation,
      sorry::engine::PlayerColor playerColor,
      int episodeIndex,
      const std::vector<std::vector<int>> &validActionsArray);

  std::pair<pybind11::object, float> getValueGradientAndValue(const std::vector<int> &observation);

  void train(std::vector<Trajectory> &&trajectories, int episodeIndex);
  void saveCheckpoint();
private:
  static constexpr float kGamma{0.9375};
  static constexpr float kPolicyNetworkLearningRate{0.001};
  static constexpr float kValueNetworkLearningRate{0.00005};
  pybind11::module jaxModule_;
  pybind11::object summaryWriter_;
  pybind11::object trainingUtilInstance_;
  pybind11::object policyOptimizer_;
  pybind11::object valueOptimizer_;
};

} // namespace python_wrapper

#endif // PYTHON_WRAPPER_REINFORCE_WITH_BASELINE_TRAINING_UTIL_HPP_