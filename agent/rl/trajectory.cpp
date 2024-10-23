#include "trajectory.hpp"

namespace py = pybind11;

Trajectory::Trajectory() {}

void Trajectory::pushStep(py::object gradient, double reward) {
  gradients.push_back(gradient);
  rewards.push_back(reward);
}

void Trajectory::reset() {
  gradients.clear();
  rewards.clear();
}