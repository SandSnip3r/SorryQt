#include "actorCriticTrainingUtil.hpp"
#include "common.hpp"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace python_wrapper {

ActorCriticTrainingUtil::ActorCriticTrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter, bool restoreFromCheckpoint) : jaxModule_(jaxModule), summaryWriter_(summaryWriter) {
  // Instantiate the MyModel class from Python
  pybind11::object TrainingUtilClass = jaxModule_.attr("TrainingUtilClass");
  // Create an instance of MyModel
  trainingUtilInstance_ = TrainingUtilClass(summaryWriter_, kPolicyNetworkLearningRate, kValueNetworkLearningRate, kCheckpointDirectoryName, restoreFromCheckpoint);
}

void ActorCriticTrainingUtil::setSeed(int seed) {
  trainingUtilInstance_.attr("setSeed")(seed);
}

std::pair<sorry::engine::Action, py::object> ActorCriticTrainingUtil::getActionAndKeyUsed(
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

float ActorCriticTrainingUtil::train(const std::vector<int> &lastObservation, float reward, const std::vector<int> &currentObservation, py::object rngKey, const std::vector<std::vector<int>> &lastValidActionsArray) {
  py::object loss = trainingUtilInstance_.attr("train")(lastObservation, reward, currentObservation, rngKey, lastValidActionsArray, kGamma);
  return loss.cast<float>();
}

void ActorCriticTrainingUtil::saveCheckpoint() {
  trainingUtilInstance_.attr("saveCheckpoint")();
}

} // namespace python_wrapper
