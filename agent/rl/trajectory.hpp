#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <pybind11/pybind11.h>

#include <vector>

class Trajectory {
public:
  Trajectory();
  void pushStep(pybind11::object gradient, double reward);
  void reset();

  std::vector<pybind11::object> gradients;
  std::vector<double> rewards;
};

#endif // TRAJECTORY_HPP_