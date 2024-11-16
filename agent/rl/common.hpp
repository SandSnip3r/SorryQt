#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <sorry/engine/card.hpp>
#include <sorry/engine/sorry.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>

namespace common {

sorry::engine::Card cardIndexToCard(size_t index);
size_t cardToCardIndex(sorry::engine::Card card);
std::vector<int> makeObservation(const sorry::engine::Sorry &sorry);
// pybind11::array_t<float> makeNumpyObservation(const sorry::engine::Sorry &sorry);

// An action in python is represented as a concatenation of 6 one-hots
// |------5------|------11-----|------67-----|------67-----|------67-----|------67-----|
// |-Action Type-|-----Card----|--Move 1 Src-|-Move 1 Dest-|--Move 2 Src-|-Move 2 Dest-|
// pybind11::array_t<float> createNumpyArrayOfActions(const std::vector<sorry::engine::Action> &actions, pybind11::module &jaxModule);
std::vector<std::vector<int>> createArrayOfActions(const std::vector<sorry::engine::Action> &actions);
sorry::engine::Action actionFromTuple(pybind11::tuple &tuple, sorry::engine::PlayerColor playerColor);

} // namespace common

#endif // COMMON_HPP_