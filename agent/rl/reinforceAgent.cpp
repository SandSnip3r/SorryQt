#include "common.hpp"
#include "reinforceAgent.hpp"

#include <pybind11/eval.h>
#include <pybind11/numpy.h>

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
  pybind11::object observation = common::makeNumpyObservation(sorry);
  std::vector<sorry::engine::Action> validActions = sorry.getActions();
  std::vector<std::vector<int>> actionsArray = common::createArrayOfActions(validActions);
  pybind11::tuple actionTuple = inferenceInstance_.attr("getBestAction")(observation, actionsArray);
  bestAction_ = common::actionFromTuple(actionTuple, sorry.getPlayerTurn());
}

std::vector<ActionScore> ReinforceAgent::getActionScores() const {
  throw std::runtime_error("Not implemented"); // TODO
  return actionScores_;
}

sorry::engine::Action ReinforceAgent::pickBestAction() const {
  if (!bestAction_.has_value()) {
    throw std::runtime_error("No best action has been calculated");
  }
  return bestAction_.value();
}

} // namespace sorry::agent