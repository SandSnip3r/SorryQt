#ifndef PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_
#define PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_

#include <sorry/engine/action.hpp>

#include <pybind11/pybind11.h>

#include <string_view>
#include <vector>

namespace python_wrapper {

class ActorCriticTrainingUtil {
public:
  ActorCriticTrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter, bool restoreFromCheckpoint);
  void setSeed(int seed);

  std::pair<sorry::engine::Action, pybind11::object> getActionAndKeyUsed(
      const std::vector<int> &observation,
      sorry::engine::PlayerColor playerColor,
      int episodeIndex,
      const std::vector<std::vector<int>> &validActionsArray);

  // Returns the value function loss.
  float train(const std::vector<int> &lastObservation, float reward, const std::vector<int> &currentObservation, pybind11::object rngKey, const std::vector<std::vector<int>> &lastValidActionsArray);
  void saveCheckpoint();
private:
  static constexpr float kGamma{0.99};
  static constexpr float kPolicyNetworkLearningRate{0.0003};
  static constexpr float kValueNetworkLearningRate{0.0003};
  static constexpr std::string_view kCheckpointDirectoryName{"latest"};
  pybind11::module jaxModule_;
  pybind11::object summaryWriter_;
  pybind11::object trainingUtilInstance_;
  pybind11::object policyOptimizer_;
  pybind11::object valueOptimizer_;
};

} // namespace python_wrapper

#endif // PYTHON_WRAPPER_ACTOR_CRITIC_TRAINING_UTIL_HPP_
