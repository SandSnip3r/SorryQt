#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <pybind11/pybind11.h>

#include <vector>


class Trajectory {
public:
  Trajectory();
  void pushStep(pybind11::object policyGradient, float reward, std::vector<int> &&observation);
  void reset();
  size_t size() const { return policyGradients.size(); }
  void setLastReward(double reward);

  // Policy gradient is directly ready for gradient descent.
  std::vector<pybind11::object> policyGradients;
  std::vector<float> rewards;
  std::vector<std::vector<int>> observations;
};

#endif // TRAJECTORY_HPP_