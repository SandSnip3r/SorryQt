#include "actionMap.hpp"
#include "common.hpp"
#include "reinforceWithBaselineTrainingUtil.hpp"

#include <pybind11/eval.h>
#include <pybind11/stl.h>

#include <iostream>

using namespace pybind11::literals;
namespace py = pybind11;

namespace python_wrapper {

ReinforceWithBaselineTrainingUtil::ReinforceWithBaselineTrainingUtil(pybind11::module jaxModule, py::object summaryWriter, std::optional<std::string> checkpointDirName) : jaxModule_(jaxModule), summaryWriter_(summaryWriter) {
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

void ReinforceWithBaselineTrainingUtil::setSeed(int seed) {
  trainingUtilInstance_.attr("setSeed")(seed);
}

std::pair<sorry::engine::Action, py::object> ReinforceWithBaselineTrainingUtil::getActionAndKeyUsed(
    const std::vector<int> &observation,
    sorry::engine::PlayerColor playerColor,
    int episodeIndex,
    const std::vector<std::vector<int>> &validActionsArray) {
  // Take an action according to the policy
  py::tuple actionAndKey = trainingUtilInstance_.attr("getActionTupleAndKeyUsed")(observation, validActionsArray);
  py::tuple actionTuple = actionAndKey[0].cast<py::tuple>();
  py::object rngKey = actionAndKey[1];
  return {common::actionFromTuple(actionTuple, playerColor), rngKey};
}

void ReinforceWithBaselineTrainingUtil::train(std::vector<Trajectory> &&trajectories, int episodeIndex) {
  // Change from an array of structs to struct of arrays
  std::vector<std::vector<float>> rewards;
  std::vector<std::vector<std::vector<int>>> observations;
  std::vector<std::vector<pybind11::object>> rngKeys;
  std::vector<std::vector<std::vector<std::vector<int>>>> validActionsArrays;
  for (Trajectory &trajectory : trajectories) {
    rewards.emplace_back(std::move(trajectory.rewards));
    observations.emplace_back(std::move(trajectory.observations));
    rngKeys.emplace_back(std::move(trajectory.rngKeys));
    validActionsArrays.emplace_back(std::move(trajectory.validActionsArrays));
  }
  trainingUtilInstance_.attr("train")(rewards, observations, rngKeys, validActionsArrays, kGamma, episodeIndex);
}

void ReinforceWithBaselineTrainingUtil::saveCheckpoint() {
  trainingUtilInstance_.attr("saveCheckpoint")();
}

} // namespace python_wrapper