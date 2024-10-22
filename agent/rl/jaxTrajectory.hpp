#ifndef JAX_TRAJECTORY_HPP_
#define JAX_TRAJECTORY_HPP_

#include <sorry/engine/card.hpp>

#include <pybind11/pybind11.h>

#include <array>

class JaxTrajectory {
public:
  JaxTrajectory(pybind11::module &jaxModule);
  void pushStep(pybind11::object gradient, double reward);
  pybind11::object getPythonTrajectory() const;
private:
  pybind11::object trajectory_;
};

#endif // JAX_TRAJECTORY_HPP_