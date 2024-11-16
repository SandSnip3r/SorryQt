#include "common.hpp"
#include "reinforceAgent.hpp"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>

using namespace pybind11::literals;

namespace sorry::agent {

ReinforceAgent::ReinforceAgent() {
  initializeJaxModule();
  pybind11::object InferenceClass = jaxModule_.attr("InferenceClass");
  inferenceInstance_ = InferenceClass();
}

ReinforceAgent::ReinforceAgent(pybind11::object trainingUtilInstance) {
  initializeJaxModule();
  pybind11::object InferenceClass = jaxModule_.attr("InferenceClass");
  inferenceInstance_ = InferenceClass(trainingUtilInstance);
}

void ReinforceAgent::initializeJaxModule() {
  // Get the sys module
  pybind11::module sys = pybind11::module::import("sys");
  // Append the directory containing my_jax.py to sys.path, SOURCE_DIR is set from CMake.
  sys.attr("path").cast<pybind11::list>().append(std::string(SOURCE_DIR));
  jaxModule_ = pybind11::module::import("jaxModule");
}

void ReinforceAgent::seed(int seed) {
  inferenceInstance_.attr("setSeed")(seed);
}

// After calling run, action preferences remain until the next call to run.
void ReinforceAgent::run(const sorry::engine::Sorry &sorry) {
  std::vector<int> observation = common::makeObservation(sorry);
  std::vector<sorry::engine::Action> validActions = sorry.getActions();
  std::vector<std::vector<int>> actionsArray = common::createArrayOfActions(validActions);
  pybind11::tuple actionTuple = inferenceInstance_.attr("getBestAction")(observation, actionsArray);
  bestAction_ = common::actionFromTuple(actionTuple, sorry.getPlayerTurn());

  // Create a temporary set of action scores and just say that we prefer the best action @ 100%.
  // TODO: Actually get real action scores from the model.
  actionScores_.clear();
  for (const auto &action : validActions) {
    actionScores_.emplace_back(action, action == bestAction_.value() ? 1.0 : 0.0);
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