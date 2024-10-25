#include "actionMap.hpp"
#include "jaxModel.hpp"

#include <iostream>

namespace py = pybind11;

JaxModel::JaxModel(py::module &jaxModule) {
  // Instantiate the MyModel class from Python
  py::object TestClass = jaxModule.attr("TestClass");
  // Create an instance of MyModel
  modelInstance_ = TestClass(ActionMap::getInstance().totalActionCount());
}

void JaxModel::setSeed(int seed) {
  modelInstance_.attr("setSeed")(seed);
}

std::pair<py::object, sorry::engine::Action> JaxModel::getGradientAndAction(
    const std::array<sorry::engine::Card, 5> &playerHand,
    const std::array<int, 4> &playerPiecePositions,
    const std::vector<sorry::engine::Action> *validActions) {
  const py::array_t<float> observation = observationToNumpyArray(playerHand, playerPiecePositions);
  py::tuple result;
  if (validActions != nullptr) {
    // Directly create the action mask for valid actions as a numpy array
    py::array_t<float> actionMask(ActionMap::getInstance().totalActionCount());
    // Initialize all values to negative infinity
    actionMask.attr("fill")(-std::numeric_limits<float>::infinity());
    for (const sorry::engine::Action &action : *validActions) {
      const int actionIndex = ActionMap::getInstance().actionToIndex(action);
      actionMask.mutable_at(actionIndex) = 0.0;
    }
    result = modelInstance_.attr("getGradientAndIndex")(observation, actionMask);
  } else {
    result = modelInstance_.attr("getGradientAndIndex")(observation);
  }
  // Take an action according to the policy
  py::object gradient = result[0];
  int index = result[1].cast<int>();
  return {gradient, ActionMap::getInstance().indexToAction(index)};
}

void JaxModel::train(const Trajectory &trajectory) {
  constexpr double kGamma = 0.99;
  constexpr double kLearningRate = 0.001;
  double returnToEnd = 0.0;
  // Iterate backward through the trajectory, calculate the return, and update the parameters.
  for (int i=trajectory.gradients.size()-1; i>=0; --i) {
    // Calculate the return
    returnToEnd = trajectory.rewards[i] + kGamma * returnToEnd;
    // Update the model parameters
    modelInstance_.attr("update")(trajectory.gradients[i], returnToEnd, kLearningRate);
  }
}

void JaxModel::saveCheckpoint() {
  modelInstance_.attr("saveCheckpoint")();
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