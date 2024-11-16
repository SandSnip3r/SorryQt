#ifndef TRAJECTORY_HPP_
#define TRAJECTORY_HPP_

#include <pybind11/pybind11.h>

#include <cstddef>
#include <vector>

class Trajectory {
public:
  Trajectory();
  void pushStep(float reward, std::vector<int> &&observation, pybind11::object rngKey, std::vector<std::vector<int>> &&validActionsArray);
  void reset();
  size_t size() const { return observations.size(); }
  void setLastReward(double reward);

  std::vector<float> rewards;
  std::vector<std::vector<int>> observations;
  std::vector<pybind11::object> rngKeys;
  std::vector<std::vector<std::vector<int>>> validActionsArrays;
};

#endif // TRAJECTORY_HPP_