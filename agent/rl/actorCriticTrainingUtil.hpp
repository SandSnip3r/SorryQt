#ifndef PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_
#define PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_

#include <sorry/engine/action.hpp>

#include <pybind11/pybind11.h>

#include <vector>

namespace python_wrapper {

class ActorCriticTrainingUtil {
public:
  ActorCriticTrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter);
  void setSeed(int seed);

  std::pair<sorry::engine::Action, py::object> getActionAndKeyUsed(
      const std::vector<int> &observation,
      sorry::engine::PlayerColor playerColor,
      int episodeIndex,
      const std::vector<std::vector<int>> &validActionsArray);

  void train(pybind11::object logProbabilityGradient, const std::vector<int> &lastObservation, const std::vector<int> &observation);
  // void saveCheckpoint();
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

#endif // PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_
