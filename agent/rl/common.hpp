#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <sorry/engine/sorry.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>

namespace common {
  
pybind11::array_t<float> makeNumpyObservation(const sorry::engine::Sorry &sorry);

} // namespace common

#endif // COMMON_HPP_