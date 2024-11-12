#include "trajectory.hpp"

namespace py = pybind11;

Trajectory::Trajectory() {}

void Trajectory::pushStep(pybind11::object policyGradient, float reward, pybind11::object valueGradient, float value) {
  policyGradients.push_back(policyGradient);
  rewards.push_back(reward);
  valueGradients.push_back(valueGradient);
  values.push_back(value);
}

void Trajectory::reset() {
  policyGradients.clear();
  rewards.clear();
  valueGradients.clear();
  values.clear();
}

void Trajectory::setLastReward(double reward) {
  if (rewards.empty()) {
    throw std::runtime_error("No rewards");
  }
  rewards.back() = reward;
}