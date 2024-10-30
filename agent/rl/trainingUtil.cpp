#include "actionMap.hpp"
#include "common.hpp"
#include "trainingUtil.hpp"

#include <iostream>

namespace py = pybind11;

namespace python_wrapper {

TrainingUtil::TrainingUtil(pybind11::module &jaxModule, std::optional<std::string> checkpointDirName) {
  // Instantiate the MyModel class from Python
  py::object TrainingUtilClass = jaxModule.attr("TrainingUtilClass");
  // Create an instance of MyModel
  if (checkpointDirName.has_value()) {
    trainingUtilInstance_ = TrainingUtilClass(ActionMap::getInstance().totalActionCount(), checkpointDirName.value());
  } else {
    trainingUtilInstance_ = TrainingUtilClass(ActionMap::getInstance().totalActionCount());
  }
}

void TrainingUtil::setSeed(int seed) {
  trainingUtilInstance_.attr("setSeed")(seed);
}

std::pair<py::object, sorry::engine::Action> TrainingUtil::getGradientAndAction(
    const sorry::engine::Sorry &sorry,
    const std::vector<sorry::engine::Action> *validActions) {
  py::object observation = common::makeNumpyObservation(sorry);
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
    result = trainingUtilInstance_.attr("getGradientAndIndex")(observation, actionMask);
  } else {
    result = trainingUtilInstance_.attr("getGradientAndIndex")(observation);
  }
  // Take an action according to the policy
  py::object gradient = result[0];
  const int index = result[1].cast<int>();
  const sorry::engine::Action action = ActionMap::getInstance().indexToActionForPlayer(index, sorry.getPlayerTurn());
  return {gradient, action};
}

void TrainingUtil::train(const Trajectory &trajectory) {
  constexpr double kGamma = 0.99;
  constexpr double kLearningRate = 0.001;
  double returnToEnd = 0.0;
  // Iterate backward through the trajectory, calculate the return, and update the parameters.
  for (int i=trajectory.gradients.size()-1; i>=0; --i) {
    // Calculate the return
    returnToEnd = trajectory.rewards[i] + kGamma * returnToEnd;
    // Update the model parameters
    trainingUtilInstance_.attr("update")(trajectory.gradients[i], returnToEnd, kLearningRate);
  }
}

void TrainingUtil::saveCheckpoint() {
  trainingUtilInstance_.attr("saveCheckpoint")();
}

} // namespace python_wrapper