#include "actionMap.hpp"
#include "common.hpp"
#include "trainingUtil.hpp"

#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include <iostream>

using namespace pybind11::literals;
namespace py = pybind11;

namespace python_wrapper {

TrainingUtil::TrainingUtil(pybind11::module jaxModule, py::object summaryWriter, std::optional<std::string> checkpointDirName) : jaxModule_(jaxModule), summaryWriter_(summaryWriter) {
  // Instantiate the MyModel class from Python
  py::object TrainingUtilClass = jaxModule_.attr("TrainingUtilClass");
  // Create an instance of MyModel
  if (checkpointDirName.has_value()) {
    trainingUtilInstance_ = TrainingUtilClass(summaryWriter_, checkpointDirName.value());
    trainingUtilInstance_.attr("loadPolicyOptimizerCheckpoint")(kPolicyNetworkLearningRate, checkpointDirName.value());
    trainingUtilInstance_.attr("loadValueOptimizerCheckpoint")(kValueNetworkLearningRate, checkpointDirName.value());
  } else {
    trainingUtilInstance_ = TrainingUtilClass(summaryWriter_);
    trainingUtilInstance_.attr("initializePolicyOptimizer")(kPolicyNetworkLearningRate);
    trainingUtilInstance_.attr("initializeValueOptimizer")(kValueNetworkLearningRate);
  }
}

void TrainingUtil::setSeed(int seed) {
  trainingUtilInstance_.attr("setSeed")(seed);
}

std::pair<py::object, sorry::engine::Action> TrainingUtil::getPolicyGradientAndAction(
    const std::vector<int> &observation,
    sorry::engine::PlayerColor playerColor,
    int episodeIndex,
    const std::vector<sorry::engine::Action> &validActions) {
  std::vector<std::vector<int>> actionsArray = common::createArrayOfActions(validActions);
  py::tuple result = trainingUtilInstance_.attr("getPolicyGradientAndActionTuple")(observation, actionsArray);
  // trainingUtilInstance_.attr("logLogitStatistics")(observation, episodeIndex);

  // Take an action according to the policy
  py::object gradient = result[0];
  py::tuple tuple = result[1].cast<py::tuple>();
  return {gradient, common::actionFromTuple(tuple, playerColor)};
}

void TrainingUtil::train(std::vector<Trajectory> &&trajectories, int episodeIndex) {
  // Change from an array of structs to struct of arrays
  std::vector<std::vector<py::object>> policyGradients;
  std::vector<std::vector<float>> rewards;
  std::vector<std::vector<std::vector<int>>> observations;
  for (Trajectory &trajectory : trajectories) {
    policyGradients.emplace_back(std::move(trajectory.policyGradients));
    rewards.emplace_back(std::move(trajectory.rewards));
    observations.emplace_back(std::move(trajectory.observations));
  }
  trainingUtilInstance_.attr("train")(policyGradients, rewards, observations, kGamma, episodeIndex);
}

void TrainingUtil::saveCheckpoint() {
  trainingUtilInstance_.attr("saveCheckpoint")();
}

} // namespace python_wrapper