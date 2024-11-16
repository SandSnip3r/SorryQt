#include "common.hpp"

#include <sorry/engine/common.hpp>

#include <pybind11/eval.h>

#include <iostream>

namespace py = pybind11;
using namespace pybind11::literals;

namespace common {

namespace {

void fillNumpyArrayWithAction(const sorry::engine::Action &action, py::array_t<float> numpyArray) {
  if (numpyArray.ndim() != 1) {
    throw std::runtime_error("Expected numpy array for action to have 1 dimension");
  }
  auto uncheckedNumpyArrayBuffer = numpyArray.mutable_unchecked<1>();

  // Write action type to buffer
  constexpr int kActionTypeBaseIndex = 0;
  switch (action.actionType) {
    case sorry::engine::Action::ActionType::kDiscard:
      uncheckedNumpyArrayBuffer(kActionTypeBaseIndex + 0) = 1.0;
      break;
    case sorry::engine::Action::ActionType::kSingleMove:
      uncheckedNumpyArrayBuffer(kActionTypeBaseIndex + 1) = 1.0;
      break;
    case sorry::engine::Action::ActionType::kDoubleMove:
      uncheckedNumpyArrayBuffer(kActionTypeBaseIndex + 2) = 1.0;
      break;
    case sorry::engine::Action::ActionType::kSorry:
      uncheckedNumpyArrayBuffer(kActionTypeBaseIndex + 3) = 1.0;
      break;
    case sorry::engine::Action::ActionType::kSwap:
      uncheckedNumpyArrayBuffer(kActionTypeBaseIndex + 4) = 1.0;
      break;
    default:
      throw std::runtime_error("Unhandled action type when converting action to numpy array");
  }

  // Write card to buffer
  constexpr int kCardBaseIndex = kActionTypeBaseIndex + 5;
  uncheckedNumpyArrayBuffer(kCardBaseIndex + common::cardToCardIndex(action.card)) = 1.0;

  if (action.move1Source < 0 || action.move1Source > 66) {
    throw std::runtime_error("Move 1 Source is out of bounds: " + std::to_string(action.move1Source));
  }
  if (action.move1Destination < 0 || action.move1Destination > 66) {
    throw std::runtime_error("Move 1 Destination is out of bounds: " + std::to_string(action.move1Destination));
  }
  if (action.move2Source < 0 || action.move2Source > 66) {
    throw std::runtime_error("Move 2 Source is out of bounds: " + std::to_string(action.move2Source));
  }
  if (action.move2Destination < 0 || action.move2Destination > 66) {
    throw std::runtime_error("Move 2 Destination is out of bounds: " + std::to_string(action.move2Destination));
  }

  // Write source & destination positions. While these values might not semantically make sense, given the action type, we don't care here.
  constexpr int kMove1SrcBaseIndex = kCardBaseIndex + 11;
  uncheckedNumpyArrayBuffer(kMove1SrcBaseIndex + action.move1Source) = 1.0;

  constexpr int kMove1DestBaseIndex = kMove1SrcBaseIndex + 67;
  uncheckedNumpyArrayBuffer(kMove1DestBaseIndex + action.move1Destination) = 1.0;

  constexpr int kMove2SrcBaseIndex = kMove1DestBaseIndex + 67;
  uncheckedNumpyArrayBuffer(kMove2SrcBaseIndex + action.move2Source) = 1.0;

  constexpr int kMove2DestBaseIndex = kMove2SrcBaseIndex + 67;
  uncheckedNumpyArrayBuffer(kMove2DestBaseIndex + action.move2Destination) = 1.0;
}

} // namespace

sorry::engine::Card cardIndexToCard(size_t index) {
  // Since the values of card enums are not a contiguous range, we can't just cast the index to a card.
  switch (index) {
    case 0: return sorry::engine::Card::kOne;
    case 1: return sorry::engine::Card::kTwo;
    case 2: return sorry::engine::Card::kThree;
    case 3: return sorry::engine::Card::kFour;
    case 4: return sorry::engine::Card::kFive;
    case 5: return sorry::engine::Card::kSeven;
    case 6: return sorry::engine::Card::kEight;
    case 7: return sorry::engine::Card::kTen;
    case 8: return sorry::engine::Card::kEleven;
    case 9: return sorry::engine::Card::kTwelve;
    case 10: return sorry::engine::Card::kSorry;
    default: throw std::runtime_error("Invalid card index");
  }
}

size_t cardToCardIndex(sorry::engine::Card card) {
  // Since the values of card enums are not a contiguous range, we can't just cast the card to an index.
  switch (card) {
    case sorry::engine::Card::kOne: return 0;
    case sorry::engine::Card::kTwo: return 1;
    case sorry::engine::Card::kThree: return 2;
    case sorry::engine::Card::kFour: return 3;
    case sorry::engine::Card::kFive: return 4;
    case sorry::engine::Card::kSeven: return 5;
    case sorry::engine::Card::kEight: return 6;
    case sorry::engine::Card::kTen: return 7;
    case sorry::engine::Card::kEleven: return 8;
    case sorry::engine::Card::kTwelve: return 9;
    case sorry::engine::Card::kSorry: return 10;
    default: throw std::runtime_error("Invalid card");
  }
}

std::vector<int> makeObservation(const sorry::engine::Sorry &sorry) {
  // Rather than do all the creation of numpy one-hots here, we'll just create a small array with integers for the observation. In JAX, we can blow them up to the right format for our models.
  std::vector<int> result;

  // -=-=- Who is playing? -=-=-
  // In order to represent who is playing, we have 3 bools, one for the player to our left, one for the player across from us, and one for the player to our right.
  // We represent other players as their position relative to us. Color does not matter.
  const std::vector<sorry::engine::PlayerColor> players = sorry.getPlayers();
  const sorry::engine::PlayerColor ourColor = sorry.getPlayerTurn();

  // const sorry::engine::PlayerColor leftColor = sorry::engine::common::rotatePlayerColor(ourColor, 1);
  // const bool haveOpponentToOurLeft = std::find(players.begin(), players.end(), leftColor) != players.end();
  // result.push_back(haveOpponentToOurLeft ? 1 : 0);

  const sorry::engine::PlayerColor acrossColor = sorry::engine::common::rotatePlayerColor(ourColor, 2);
  const bool haveOpponentAcross = std::find(players.begin(), players.end(), acrossColor) != players.end();
  result.push_back(haveOpponentAcross ? 1 : 0);

  // const sorry::engine::PlayerColor rightColor = sorry::engine::common::rotatePlayerColor(ourColor, 3);
  // const bool haveOpponentToOurRight = std::find(players.begin(), players.end(), rightColor) != players.end();
  // result.push_back(haveOpponentToOurRight ? 1 : 0);

  // -=-=- What cards do we have? -=-=-
  const std::array<sorry::engine::Card, 5> playerHand = sorry.getHandForPlayer(ourColor);
  for (sorry::engine::Card card : playerHand) {
    result.push_back(common::cardToCardIndex(card));
  }

  auto addPiecePositionsForPlayer = [&](sorry::engine::PlayerColor playerColor) {
    const std::array<int, 4> piecePositions = sorry.getPiecePositionsForPlayer(playerColor);
    for (int position : piecePositions) {
      result.push_back(position);
    }
  };

  // -=-=- Where are the pieces on the board? -=-=-
  addPiecePositionsForPlayer(ourColor);

  // if (haveOpponentToOurLeft) {
  //   addPiecePositionsForPlayer(leftColor);
  // } else {
  //   // If we don't have an opponent to our left, we'll just add 4 zeros.
  //   result.insert(result.end(), 4, 0);
  // }

  if (haveOpponentAcross) {
    addPiecePositionsForPlayer(acrossColor);
  } else {
    // If we don't have an opponent across from us, we'll just add 4 zeros.
    result.insert(result.end(), 4, 0);
  }

  // if (haveOpponentToOurRight) {
  //   addPiecePositionsForPlayer(rightColor);
  // } else {
  //   // If we don't have an opponent to our right, we'll just add 4 zeros.
  //   result.insert(result.end(), 4, 0);
  // }

  // TODO: Add the discarded cards to the observation.
  return result;
}

py::array_t<float> makeNumpyObservation(const sorry::engine::Sorry &sorry) {
  std::array<sorry::engine::Card, 5> playerHand = sorry.getHandForPlayer(sorry.getPlayerTurn());
  std::array<int, 4> ourPiecePositions = sorry.getPiecePositionsForPlayer(sorry.getPlayerTurn());
  std::array<int, 4> opponentPiecePositions = sorry.getPiecePositionsForPlayer(sorry::engine::PlayerColor::kBlue); // TODO: Blue is hard-coded for now.
  // TODO: For now, we're not going to bother with the discarded cards.
  // std::vector<sorry::engine::Card> discardedCards = sorry.getDiscardedCards();

  constexpr std::size_t playerColorCount = 4;
  constexpr std::size_t handCardCount = std::decay_t<decltype(playerHand)>().size();
  constexpr std::size_t cardTypeCount = 11;
  constexpr std::size_t playerCount = 2;
  constexpr std::size_t pieceCount = std::decay_t<decltype(ourPiecePositions)>().size();
  constexpr std::size_t piecePositionTypeCount = 67;

  // Declare a numpy array for the entire observation.
  // The version below does not work on 2.9.1-2 but does on 2.11.1-2, it seems to result in a memory layout which does not allow for slicing views.
  py::array_t<float> numpyObservation(playerColorCount + handCardCount * cardTypeCount + playerCount * pieceCount * piecePositionTypeCount);
  // The version below worked when I was using pybind11 2.9.1-2.
  // py::array_t<float> numpyObservation({handCardCount * cardTypeCount + pieceCount * piecePositionTypeCount}, {sizeof(float)});

  // std::cout << "Observation size: " << numpyObservation.size() << std::endl;
  // printBuffer(numpyObservation);

  // Initialize to all 0s, since we're filling a lot of one-hot vectors; most elements will be 0.
  numpyObservation.attr("fill")(0.0);

  // This part of the observation contains the current player's color. We'll use a one-hot encoding of the player color.
  py::slice playerColorSliceDims(0, playerColorCount, 1);
  py::array_t<float> playerColorSlice = py::cast<py::array_t<float>>(numpyObservation[playerColorSliceDims]);
  auto uncheckedPlayerColorBuffer = playerColorSlice.mutable_unchecked<1>();
  uncheckedPlayerColorBuffer(static_cast<int>(sorry.getPlayerTurn())) = 1.0;

  // This part of the observation contains the player's hand. We'll use a one-hot encoding of the card type for the 5 cards.
  py::slice handSliceDims(playerColorCount, playerColorCount + handCardCount*cardTypeCount, 1);
  py::array handSlice = numpyObservation[handSliceDims];
  py::array_t<float> reshapedHandSlice = py::cast<py::array_t<float>>(handSlice.attr("reshape")(handCardCount, cardTypeCount));
  auto uncheckedHandBuffer = reshapedHandSlice.mutable_unchecked<2>();
  for (size_t cardIndex=0; cardIndex<playerHand.size(); ++cardIndex) {
    uncheckedHandBuffer(cardIndex, common::cardToCardIndex(playerHand[cardIndex])) = 1.0;
  }

  // This part of the observation contains the player's piece positions. We'll use a one-hot encoding of the position type for the 4 pieces.
  constexpr std::size_t baseIndex = playerColorCount + handCardCount*cardTypeCount;
  py::slice pieceSliceDims(baseIndex, baseIndex + playerCount*pieceCount*piecePositionTypeCount, 1);
  py::array pieceSlice = numpyObservation[pieceSliceDims];
  py::array_t<float> reshapedPieceSlice = py::cast<py::array_t<float>>(pieceSlice.attr("reshape")(playerCount*pieceCount, piecePositionTypeCount));
  auto uncheckedPiecesBuffer = reshapedPieceSlice.mutable_unchecked<2>();
  for (size_t pieceIndex=0; pieceIndex<ourPiecePositions.size(); ++pieceIndex) {
    uncheckedPiecesBuffer(pieceIndex, ourPiecePositions[pieceIndex]) = 1.0;
  }
  for (size_t pieceIndex=0; pieceIndex<opponentPiecePositions.size(); ++pieceIndex) {
    uncheckedPiecesBuffer(pieceIndex+4, opponentPiecePositions[pieceIndex]) = 1.0;
  }

  return numpyObservation;
}

py::array_t<float> createNumpyArrayOfActions(const std::vector<sorry::engine::Action> &actions, pybind11::module &jaxModule) {
  constexpr int kActionSize = 5 + 11 + 67 + 67 + 67 + 67;
  // TODO: Maybe in the future when we want to pad the actions array, we can do it here.
  py::array_t<float> array(py::array::ShapeContainer({static_cast<int>(actions.size()), kActionSize}));
  array.attr("fill")(0.0);
  auto locals = pybind11::dict("actionsArray"_a=array, "actionSize"_a=kActionSize);
  for (int actionIndex=0; actionIndex<actions.size(); ++actionIndex) {
    locals["actionIndex"] = actionIndex;
    py::object res = pybind11::eval("actionsArray[actionIndex:actionIndex+1, :].reshape((actionSize,))", jaxModule.attr("__dict__"), locals);
    fillNumpyArrayWithAction(actions[actionIndex], py::cast<py::array_t<float>>(res));
  }
  return array;
}

std::vector<std::vector<int>> createArrayOfActions(const std::vector<sorry::engine::Action> &actions) {
  std::vector<std::vector<int>> result;
  for (const sorry::engine::Action &action : actions) {
    std::vector<int> actionVector(6);
    actionVector[0] = static_cast<int>(action.actionType);
    actionVector[1] = cardToCardIndex(action.card);
    actionVector[2] = action.move1Source;
    actionVector[3] = action.move1Destination;
    actionVector[4] = action.move2Source;
    actionVector[5] = action.move2Destination;
    result.emplace_back(std::move(actionVector));
  }
  return result;
}

sorry::engine::Action actionFromTuple(py::tuple &tuple, sorry::engine::PlayerColor playerColor) {
  return sorry::engine::Action(
      playerColor,
      static_cast<sorry::engine::Action::ActionType>(tuple[0].cast<int>()),
      common::cardIndexToCard(tuple[1].cast<int>()),
      tuple[2].cast<int>(),
      tuple[3].cast<int>(),
      tuple[4].cast<int>(),
      tuple[5].cast<int>()
      );
}

} // namespace common