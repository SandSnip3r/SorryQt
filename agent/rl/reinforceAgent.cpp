#include "actionMap.hpp"
#include "common.hpp"
#include "reinforceAgent.hpp"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <string>

using namespace pybind11::literals;

namespace sorry::agent {

ReinforceAgent::ReinforceAgent() {
  // Get the sys module
  pybind11::module sys = pybind11::module::import("sys");
  // Append the directory containing my_jax.py to sys.path, SOURCE_DIR is set from CMake.
  sys.attr("path").cast<pybind11::list>().append(std::string(SOURCE_DIR));
  jaxModule_ = pybind11::module::import("jaxModule");
  pybind11::object InferenceClass = jaxModule_.attr("InferenceClass");
  inferenceInstance_ = InferenceClass(ActionMap::getInstance().totalActionCount());
}

void ReinforceAgent::seed(int seed) {
  auto locals = pybind11::dict("seed"_a=seed);
  jaxRandomKey_ = pybind11::eval("jax.random.key(seed)", jaxModule_.attr("__dict__"), locals);
}

pybind11::object ReinforceAgent::getRandomKey() {
  auto locals = pybind11::dict("key"_a=jaxRandomKey_);
  pybind11::tuple keys = pybind11::eval("jax.random.split(key)", jaxModule_.attr("__dict__"), locals);
  jaxRandomKey_ = keys[0];
  return keys[1];
}

// After calling run, action preferences remain until the next call to run.
void ReinforceAgent::run(const sorry::engine::Sorry &sorry) {
  // Create the observation
  pybind11::object observation = common::makeNumpyObservation(sorry);

  // Create the action mask for valid actions as a numpy array
  pybind11::array_t<float> actionMask(ActionMap::getInstance().totalActionCount());
  // Initialize all values to negative infinity
  actionMask.attr("fill")(-std::numeric_limits<float>::infinity());
  const std::vector<sorry::engine::Action> validActions = sorry.getActions();
  for (const sorry::engine::Action &action : validActions) {
    const int actionIndex = ActionMap::getInstance().actionToIndex(action);
    actionMask.mutable_at(actionIndex) = 0.0;
  }

  pybind11::tuple probabilitiesAndIndex = inferenceInstance_.attr("getProbabilitiesAndSelectedIndex")(observation, actionMask);
  if (probabilitiesAndIndex.size() != 2) {
    throw std::runtime_error("Expected two values from getProbabilitiesAndSelectedIndex");
  }
  pybind11::array_t<float> probabilities = pybind11::cast<pybind11::array_t<float>>(probabilitiesAndIndex[0]);
  int actionIndex = pybind11::cast<int>(probabilitiesAndIndex[1]);
  
  // Save best action
  bestAction_ = ActionMap::getInstance().indexToActionForPlayer(actionIndex, sorry.getPlayerTurn());

  // Reset action probabilities
  actionScores_.clear();
  for (const sorry::engine::Action &action : validActions) {
    int index = ActionMap::getInstance().actionToIndex(action);
    if (index >= probabilities.size()) {
      throw std::runtime_error("Index out of range");
    }
    actionScores_.emplace_back(action, probabilities.at(index));
  }
}

std::vector<ActionScore> ReinforceAgent::getActionScores() const {
  return actionScores_;
}

sorry::engine::Action ReinforceAgent::pickBestAction() const {
  if (!bestAction_.has_value()) {
    throw std::runtime_error("No best action has been calculated");
  }
  return bestAction_.value();
}

} // namespace sorry::agent