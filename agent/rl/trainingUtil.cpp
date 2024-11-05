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
    trainingUtilInstance_ = TrainingUtilClass(ActionMap::getInstance().totalActionCount(), summaryWriter_, checkpointDirName.value());
  } else {
    trainingUtilInstance_ = TrainingUtilClass(ActionMap::getInstance().totalActionCount(), summaryWriter_);
  }

  trainingUtilInstance_.attr("initializePolicyOptimizer")(kPolicyNetworkLearningRate);
  trainingUtilInstance_.attr("initializeValueOptimizer")(kValueNetworkLearningRate);
}

void TrainingUtil::setSeed(int seed) {
  trainingUtilInstance_.attr("setSeed")(seed);
}

std::pair<py::object, sorry::engine::Action> TrainingUtil::getPolicyGradientAndAction(
    pybind11::object observation,
    sorry::engine::PlayerColor playerColor,
    int episodeIndex,
    const std::vector<sorry::engine::Action> *validActions) {
  py::tuple result;
  if (validActions != nullptr) {
    // Directly create the action mask for valid actions as a numpy array
    py::array_t<float> actionMask(ActionMap::getInstance().totalActionCount());
    // Initialize all values to negative infinity
    actionMask.attr("fill")(-std::numeric_limits<float>::infinity());
    for (const sorry::engine::Action &action : *validActions) {
      const int actionIndex = ActionMap::getInstance().actionToIndex(action);
      actionMask.mutable_at(actionIndex) = 0.0;
    }
    result = trainingUtilInstance_.attr("getPolicyGradientAndIndex")(observation, actionMask);
  } else {
    result = trainingUtilInstance_.attr("getPolicyGradientAndIndex")(observation);
  }
  // trainingUtilInstance_.attr("logLogitStatistics")(observation, episodeIndex);

  // Take an action according to the policy
  py::object gradient = result[0];
  const int index = result[1].cast<int>();
  const sorry::engine::Action action = ActionMap::getInstance().indexToActionForPlayer(index, playerColor);
  return {gradient, action};
}

std::pair<pybind11::object, float> TrainingUtil::getValueGradientAndValue(pybind11::object observation) {
  py::tuple result = trainingUtilInstance_.attr("getValueGradientAndValue")(observation);
  py::object gradient = result[0];
  const float value = result[1].cast<float>();
  return {gradient, value};
}

void TrainingUtil::train(std::vector<Trajectory> &trajectories, int episodeIndex) {
  // Change from an array of structs to struct of arrays
  std::vector<std::vector<py::object>> policyGradients;
  std::vector<std::vector<float>> rewards;
  std::vector<std::vector<py::object>> valueGradients;
  std::vector<std::vector<float>> values;
  for (Trajectory &trajectory : trajectories) {
    policyGradients.push_back(std::move(trajectory.policyGradients));
    rewards.push_back(std::move(trajectory.rewards));
    valueGradients.push_back(std::move(trajectory.valueGradients));
    values.push_back(std::move(trajectory.values));
  }
  trainingUtilInstance_.attr("train")(policyGradients, valueGradients, rewards, values, kGamma, episodeIndex);
}

void TrainingUtil::trainOld(const Trajectory &trajectory, int episodeIndex) {
  constexpr float kGamma = 0.99;
  constexpr float kL2Regularization = 0.01;
  std::vector<float> tdErrors;
  tdErrors.reserve(trajectory.size());

  // Iterate backward through the trajectory, calculate the return, and update the parameters.
  for (int i=0; i<trajectory.size(); ++i) {
    // Calculate the return
    float returnToEnd = 0.0;
    for (int k=i+1; k<trajectory.size(); ++k) {
      returnToEnd += trajectory.rewards[k] + std::pow(kGamma, k-i-1);
    }
    // returnToEnd = trajectory.rewards[i] + kGamma * returnToEnd;
    // Calculate the td-error
    const float tdError = returnToEnd - trajectory.values[i];
    tdErrors.push_back(tdError);
    // std::cout << "TD Error: " << tdError << std::endl;

    // Scale the gradients
    auto locals = pybind11::dict("policyGradient"_a=trajectory.policyGradients[i],
                                 "valueGradient"_a=trajectory.valueGradients[i],
                                 "tdError"_a=tdError,
                                 "gamma"_a=kGamma,
                                 "i"_a=i);
    py::exec(R"(
      scaledPolicyGradient = jax.tree_util.tree_map(lambda x, tdError=tdError, gamma=gamma, i=i: x * tdError * gamma**i, policyGradient)
      scaledValueGradient = jax.tree_util.tree_map(lambda x, tdError=tdError: x * tdError, valueGradient)
    )", jaxModule_.attr("__dict__"), locals);
    py::object scaledPolicyGradient = locals["scaledPolicyGradient"];
    py::object scaledValueGradient = locals["scaledValueGradient"];

    // Update the model parameters
    policyOptimizer_.attr("update")(scaledPolicyGradient);
    valueOptimizer_.attr("update")(scaledValueGradient);
  }

  float meanTdError = std::accumulate(tdErrors.begin(), tdErrors.end(), 0.0f) / tdErrors.size();
  float sumSquaredDiffs = std::accumulate(tdErrors.begin(), tdErrors.end(), 0.0f, [meanTdError](float sum, float tdError) {
    return sum + std::pow(tdError - meanTdError, 2);
  });
  float stddevTdError = std::sqrt(sumSquaredDiffs / tdErrors.size());
  summaryWriter_.attr("add_scalar")("episode/tdErrorMean", meanTdError, episodeIndex);
  summaryWriter_.attr("add_scalar")("episode/tdErrorStdDev", stddevTdError, episodeIndex);
}

void TrainingUtil::saveCheckpoint() {
  trainingUtilInstance_.attr("saveCheckpoint")();
}

} // namespace python_wrapper