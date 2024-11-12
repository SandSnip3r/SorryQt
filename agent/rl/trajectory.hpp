#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <pybind11/pybind11.h>

#include <vector>

class Trajectory {
public:
  Trajectory();
  void pushStep(pybind11::object policyGradient, float reward, pybind11::object valueGradient, float value);
  void reset();
  size_t size() const { return policyGradients.size(); }
  void setLastReward(double reward);

  // Policy gradient is directly ready for gradient descent.
  std::vector<pybind11::object> policyGradients;
  std::vector<float> rewards;
  // Value gradient is based on gradient ascent, make sure to negate the gradient before passing to a gradient descent optimizer.
  std::vector<pybind11::object> valueGradients;
  std::vector<float> values;
};

#endif // TRAJECTORY_HPP_