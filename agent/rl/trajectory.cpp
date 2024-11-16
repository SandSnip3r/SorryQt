#include "trajectory.hpp"

#include <stdexcept>

Trajectory::Trajectory() {}

void Trajectory::pushStep(float reward, std::vector<int> &&observation, pybind11::object rngKey, std::vector<std::vector<int>> &&validActionsArray) {
  rewards.push_back(reward);
  observations.emplace_back(std::move(observation));
  rngKeys.push_back(rngKey);
  validActionsArrays.emplace_back(std::move(validActionsArray));
}

void Trajectory::reset() {
  rewards.clear();
  observations.clear();
  rngKeys.clear();
  validActionsArrays.clear();
}

void Trajectory::setLastReward(double reward) {
  if (rewards.empty()) {
    throw std::runtime_error("No rewards");
  }
  rewards.back() = reward;
}