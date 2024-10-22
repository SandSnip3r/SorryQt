#include "actionMap.hpp"
#include "jaxModel.hpp"

#include <iostream>

namespace py = pybind11;

JaxModel::JaxModel(py::module &jaxModule) {
  // Instantiate the MyModel class from Python
  py::object TestClass = jaxModule.attr("TestClass");
  // Create an instance of MyModel
  modelInstance_ = TestClass();
}

void JaxModel::setSeed(int seed) {
  modelInstance_.attr("setSeed")(seed);
}

std::pair<py::object, sorry::engine::Action> JaxModel::getGradientAndAction(const std::array<sorry::engine::Card, 5> &playerHand, const std::array<int, 4> &playerPiecePositions) {
  const py::array_t<float> numpyObservation = observationToNumpyArray(playerHand, playerPiecePositions);
  // Take an action according to the policy
  py::tuple res = modelInstance_.attr("getGradientAndIndex")(numpyObservation);
  py::object gradient = res[0];
  int index = res[1].cast<int>();
  return {gradient, ActionMap::getInstance().indexToAction(index)};
}

void JaxModel::train(const JaxTrajectory &trajectory) {
  modelInstance_.attr("train")(trajectory.getPythonTrajectory());
}

py::array_t<float> JaxModel::observationToNumpyArray(const std::array<sorry::engine::Card, 5> &playerHand,
                                          const std::array<int, 4> &playerPiecePositions) const {
  auto printBuffer = [](const auto &buffer) {
    const auto info = buffer.request();
    std::cout << "Buffer info: " << std::endl;
    std::cout << "  ptr: " << info.ptr << std::endl;
    std::cout << "  itemsize: " << info.itemsize << std::endl;
    std::cout << "  size: " << info.size << std::endl;
    std::cout << "  ndim: " << info.ndim << std::endl;
    std::cout << "  shape: [ ";
    for (size_t i=0; i<info.shape.size(); ++i) {
      std::cout << info.shape[i] << " ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  strides: [ ";
    for (size_t i=0; i<info.strides.size(); ++i) {
      std::cout << info.strides[i] << " ";
    }
    std::cout << "]" << std::endl;
  };

  constexpr std::size_t handCardCount = std::decay_t<decltype(playerHand)>().size();
  constexpr std::size_t cardTypeCount = 11;
  constexpr std::size_t pieceCount = std::decay_t<decltype(playerPiecePositions)>().size();
  constexpr std::size_t piecePositionTypeCount = 67; // TODO: This will need to change once we have more than one player.

  // Declare a numpy array for the entire observation.
  // The version below does not work on 2.9.1-2 but does on 2.11.1-2, it seems to result in a memory layout which does not allow for slicing views.
  py::array_t<float> numpyObservation(handCardCount * cardTypeCount + pieceCount * piecePositionTypeCount);
  // The version below worked when I was using pybind11 2.9.1-2.
  // py::array_t<float> numpyObservation({handCardCount * cardTypeCount + pieceCount * piecePositionTypeCount}, {sizeof(float)});

  // printBuffer(numpyObservation);

  // Initialize to all 0s, since we're filling a lot of one-hot vectors; most elements will be 0.
  numpyObservation.attr("fill")(0.0);

  // The first part of the observation contains the player's hand. We'll use a one-hot encoding of the card type for the 5 cards.
  py::slice handSliceDims(0, handCardCount*cardTypeCount, 1);
  py::array handSlice = numpyObservation[handSliceDims];
  py::array_t<float> reshapedHandSlice = py::cast<py::array_t<float>>(handSlice.attr("reshape")(handCardCount, cardTypeCount));
  auto uncheckedHandBuffer = reshapedHandSlice.mutable_unchecked<2>();
  for (size_t cardIndex=0; cardIndex<playerHand.size(); ++cardIndex) {
    uncheckedHandBuffer(cardIndex, static_cast<int>(playerHand[cardIndex])) = 1.0;
  }

  // The second part of the observation contains the player's piece positions. We'll use a one-hot encoding of the position type for the 4 pieces.
  py::slice pieceSliceDims(handCardCount*cardTypeCount, handCardCount*cardTypeCount + pieceCount*piecePositionTypeCount, 1);
  py::array pieceSlice = numpyObservation[pieceSliceDims];
  py::array_t<float> reshapedPieceSlice = py::cast<py::array_t<float>>(pieceSlice.attr("reshape")(pieceCount, piecePositionTypeCount));
  auto uncheckedPiecesBuffer = reshapedPieceSlice.mutable_unchecked<2>();
  for (size_t pieceIndex=0; pieceIndex<playerPiecePositions.size(); ++pieceIndex) {
    uncheckedPiecesBuffer(pieceIndex, playerPiecePositions[pieceIndex]) = 1.0;
  }

  return numpyObservation;
}