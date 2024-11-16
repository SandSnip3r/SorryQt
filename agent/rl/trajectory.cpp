#include "trajectory.hpp"

namespace py = pybind11;

Trajectory::Trajectory() {}

void Trajectory::pushStep(pybind11::object policyGradient, float reward, std::vector<int> &&observation) {
  policyGradients.push_back(policyGradient);
  rewards.push_back(reward);
  observations.emplace_back(std::move(observation));
}

void Trajectory::reset() {
  policyGradients.clear();
  rewards.clear();
  observations.clear();
}

void Trajectory::setLastReward(double reward) {
  if (rewards.empty()) {
    throw std::runtime_error("No rewards");
  }
  rewards.back() = reward;
}