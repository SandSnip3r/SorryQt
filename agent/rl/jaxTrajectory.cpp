#include "jaxTrajectory.hpp"

#include <iostream>

namespace py = pybind11;

JaxTrajectory::JaxTrajectory(py::module &jaxModule) {
  // Instantiate the Trajectory class from Python
  py::object Trajectory = jaxModule.attr("Trajectory");
  // Create an instance of MyModel
  trajectory_ = Trajectory();
}

void JaxTrajectory::pushStep(pybind11::object gradient, double reward) {
  trajectory_.attr("pushStep")(gradient, reward);
}

pybind11::object JaxTrajectory::getPythonTrajectory() const {
  return trajectory_;
}