#include "jaxModel.hpp"

#include <sorry/common/common.hpp>
#include <sorry/engine/sorry.hpp>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <random>

/* General TODOs:
** - Sorry::reset() method
**/

using namespace std;

// How long does it take an agent acting randomly to finish a game of Sorry?
void simulateRandomGames() {
  constexpr const int kNumGames = 1'000'000;
  int totalActionCount = 0;
  mt19937 randomEngine(123);// = sorry::common::createRandomEngine();
  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kGreen});
  for (int i=0; i<kNumGames; ++i) {
    int thisGameActionCount = 0;
    sorry.reset(randomEngine);
    while (!sorry.gameDone()) {
      const std::vector<sorry::engine::Action> actions = sorry.getActions();
      uniform_int_distribution<size_t> dist(0, actions.size() - 1);
      const sorry::engine::Action action = actions[dist(randomEngine)];
      sorry.doAction(action, randomEngine);
      ++thisGameActionCount;
    }
    totalActionCount += thisGameActionCount;
    if (i%10000 == 0) {
      cout << i << ".." << flush;
      // cout << "Game " << i << ". Average actions per game: " << static_cast<double>(totalActionCount) / (i+1) << endl;
    }
    if (i%100000 == 0) {
      cout << endl;
    }
  }
  double avg = static_cast<double>(totalActionCount) / kNumGames;
  cout << endl << "Average actions per game: " << avg << endl;
}

namespace py = pybind11;

void trainReinforce(py::module &jaxModule) {
  // Initialize python module/model
  JaxModel model(jaxModule);

  // Seed all random engines
  constexpr int kSeed = 0x5EED;
  std::mt19937 randomEngine{kSeed};
  model.setSeed(kSeed);

  // Construct Sorry game
  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kGreen});
  // TODO: Color shouldn't matter, I just randomly picked green.

  constexpr int kEpisodeCount = 1;
  // constexpr int kEpisodeCount = 10'000;
  for (int i=0; i<kEpisodeCount; ++i) {
    // Generate a full trajectory according to the policy
    sorry.reset(randomEngine);
    JaxTrajectory trajectory;
    while (!sorry.gameDone()) {
      cout << "Game state: " << sorry.toString() << endl;
      // Get the current observation
      std::array<sorry::engine::Card, 5> playerHand = sorry.getHandForPlayer(sorry.getPlayerTurn());
      std::array<int, 4> playerPiecePositions = sorry.getPiecePositionsForPlayer(sorry.getPlayerTurn());
      // TODO: For now, we're not going to bother with the discarded cards.
      // std::vector<sorry::engine::Card> discardedCards = sorry.getDiscardedCards();

      // Take an action according to the policy
      sorry::engine::Action action = model.getAction(playerHand, playerPiecePositions);

      // For now, we terminate the episode if the action is invalid
      std::vector<sorry::engine::Action> actions = sorry.getActions();
      if (std::find(actions.begin(), actions.end(), action) == actions.end()) {
        // This action is invalid, terminate the episode.
        cout << "Invalid action: " << action.toString() << endl;
        break;
      }

      // Take action in game
      cout << "Taking action: " << action.toString() << endl;
      sorry.doAction(action, randomEngine);

      float reward;
      if (sorry.gameDone()) {
        reward = 0.0;
      } else {
        // Give a small negative reward to encourage the model to finish the game as quickly as possible
        reward = -0.1;
      }

      // Store the observation into a python-read trajectory data structure
      trajectory.pushStep(playerHand, playerPiecePositions, reward);
    }
    // TODO(GPT): A full trajectory has been generated, now train the model
    model.train(trajectory);
  }
}

int main() {
  constexpr int kSeed = 0x5EED;
  mt19937 randomEngine{kSeed};

  // Initialize the Python interpreter
  py::scoped_interpreter guard{};

  // Get the sys module
  py::module sys = py::module::import("sys");

  // Append the directory containing my_jax.py to sys.path, SOURCE_DIR is set from CMake.
  sys.attr("path").cast<py::list>().append(std::string(SOURCE_DIR));

  // Load the Python module
  py::module jax_module = py::module::import("my_jax");

  trainReinforce(jax_module);

  // // Instantiate the MyModel class from Python
  // py::object TestClass = jax_module.attr("TestClass");

  // // Create an instance of MyModel
  // py::object model_instance = TestClass();

  // // ============ Quick test ============
  // sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kYellow});
  // sorry.reset(randomEngine);
  // std::array<sorry::engine::Card, 5> playerHand = sorry.getHandForPlayer(sorry.getPlayerTurn());
  // std::array<int, 4> playerPiecePositions = sorry.getPiecePositionsForPlayer(sorry.getPlayerTurn());
  // const py::array_t<float> numpyObservation = observationToNumpyArray(playerHand, playerPiecePositions);
  // cout << "Consulting model on board: " << sorry.toString() << endl;
  // py::object res = model_instance.attr("getAction")(numpyObservation);
  // sorry::engine::Action action = numpyArrayToAction(py::cast<py::array_t<float>>(res));
  // cout << "Model chose action: " << action.toString() << endl;
  // return 0;
  // // ====================================

  // // Call function of class
  // constexpr int kNumIterations = 10000;
  // std::array<float, kNumIterations> results;
  // std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  // for (int i=0; i<kNumIterations; ++i) {
  //   const py::object result = model_instance.attr("func")(i);
  //   const auto &numpy_array = result.cast<py::array_t<float>>();
  //   const auto &data = numpy_array.unchecked<1>();
  //   results[i] = data(0);
  // }
  // std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  // std::cout << "Results: [";
  // for (int i=0; i<kNumIterations; ++i) {
  //   std::cout << results[i] << ", ";
  // }
  // std::cout << "]" << std::endl;
  // std::cout << "Calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

  return 0;
}