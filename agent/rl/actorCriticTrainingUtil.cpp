#include "actorCriticTrainingUtil.hpp"
#include "common.hpp"

namespace python_wrapper {

ActorCriticTrainingUtil::ActorCriticTrainingUtil(pybind11::module jaxModule, pybind11::object summaryWriter) : jaxModule_(jaxModule), summaryWriter_(summaryWriter) {
  // Instantiate the MyModel class from Python
  pybind11::object TrainingUtilClass = jaxModule_.attr("TrainingUtilClass");
  // Create an instance of MyModel
  trainingUtilInstance_ = TrainingUtilClass(summaryWriter_);
  trainingUtilInstance_.attr("initializePolicyOptimizer")(kPolicyNetworkLearningRate);
  trainingUtilInstance_.attr("initializeValueOptimizer")(kValueNetworkLearningRate);
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

void ActorCriticTrainingUtil::train(pybind11::object logProbabilityGradient, const std::vector<int> &lastObservation, const std::vector<int> &observation) {
  // TODO
}

} // namespace python_wrapper
