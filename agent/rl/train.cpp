#include "actionMap.hpp"
#include "common.hpp"
#include "trainingUtil.hpp"
#include "trajectory.hpp"

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

void trainReinforce() {
  // Load the Python module
  py::module jaxModule = py::module::import("jaxModule");
  py::module tensorboardX = py::module::import("tensorboardX");
  py::object summaryWriter = tensorboardX.attr("SummaryWriter")();

  constexpr bool kUseActionMasking{true};

  // Initialize python module/model
  python_wrapper::TrainingUtil pythonTrainingUtil(jaxModule, "latest");

  // Seed all random engines
  constexpr int kSeed = 0x5EED;
  std::mt19937 randomEngine{kSeed};
  pythonTrainingUtil.setSeed(kSeed);

  auto pickRandomColor = [&randomEngine]() {
    const std::array<sorry::engine::PlayerColor, 4> colors = {
      sorry::engine::PlayerColor::kGreen,
      sorry::engine::PlayerColor::kYellow,
      sorry::engine::PlayerColor::kRed,
      sorry::engine::PlayerColor::kBlue
    };
    std::uniform_int_distribution<size_t> dist(0, colors.size()-1);
    return colors[dist(randomEngine)];
  };

  Trajectory trajectory;

  constexpr int kEpisodeCount = 1'000'000;
  for (int i=0; i<kEpisodeCount; ++i) {
    auto episodeStartTime = std::chrono::high_resolution_clock::now();
    // Construct Sorry game
    // TODO: This construction should be moved outside the loop
    const sorry::engine::PlayerColor playerColor = pickRandomColor();
    sorry::engine::Sorry sorry({playerColor});
    sorry.reset(randomEngine);
    trajectory.reset();

    // Generate a full trajectory according to the policy
    int actionCount = 0;
    while (!sorry.gameDone()) {
      const std::vector<sorry::engine::Action> actions = sorry.getActions();
      py::object gradient;
      sorry::engine::Action action;
      if constexpr (kUseActionMasking) {
        // Take an action according to the policy, masked by the valid actions
        std::tie(gradient, action) = pythonTrainingUtil.getGradientAndAction(sorry, &actions);

        if (std::find(actions.begin(), actions.end(), action) == actions.end()) {
          std::cout << "Current state: " << sorry.toString() << std::endl;
          std::cout << "Valid actions were:" << std::endl;
          for (const sorry::engine::Action &a : actions) {
            std::cout << "  " << a.toString() << std::endl;
          }
          throw std::runtime_error("Invalid action after mask "+action.toString());
        }
      } else {
        std::tie(gradient, action) = pythonTrainingUtil.getGradientAndAction(sorry);

        // Terminate the episode if the action is invalid
        if (std::find(actions.begin(), actions.end(), action) == actions.end()) {
          // This action is invalid, terminate the episode.
          trajectory.pushStep(gradient, -1.0);
          break;
        }
      }

      // Take action in game
      sorry.doAction(action, randomEngine);
      ++actionCount;

      float reward;
      if (sorry.gameDone()) {
        reward = 0.0;
      } else {
        // Give a small negative reward to encourage the model to finish the game as quickly as possible
        reward = -0.01;
      }

      // Store the observation into a python-read trajectory data structure
      trajectory.pushStep(gradient, reward);
    }
    // A full trajectory has been generated, now train the model
    auto trainStartTime = std::chrono::high_resolution_clock::now();
    pythonTrainingUtil.train(trajectory);
    auto endTime = std::chrono::high_resolution_clock::now();
    // std::cout << "Episode #" << i << " took " << actionCount << " actions, " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime-trainStartTime).count() << "ms to train, and " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime-episodeStartTime).count() << "ms total" << std::endl;
    summaryWriter.attr("add_scalar")("episode/action_count", actionCount, i);

    if ((i+1)%100 == 0) {
      cout << "Episode " << i << " complete" << endl;
      if ((i+1)%1000 == 0) {
        pythonTrainingUtil.saveCheckpoint();
      }
    }
  }
}

void loadModel() {
  py::module jaxModule = py::module::import("jaxModule");
  py::object InferenceClass = jaxModule.attr("InferenceClass");
  py::object inferenceInstance = InferenceClass(ActionMap::getInstance().totalActionCount());

  // Seed all random engines
  constexpr int kSeed = 0x5EED;
  std::mt19937 randomEngine{kSeed};
  inferenceInstance.attr("setSeed")(kSeed);

  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kGreen});
  sorry.reset(randomEngine);
  while (!sorry.gameDone()) {
    // Create the observation
    py::object observation = common::makeNumpyObservation(sorry);

    // Create the action mask for valid actions as a numpy array
    py::array_t<float> actionMask(ActionMap::getInstance().totalActionCount());
    // Initialize all values to negative infinity
    actionMask.attr("fill")(-std::numeric_limits<float>::infinity());
    const std::vector<sorry::engine::Action> validActions = sorry.getActions();
    for (const sorry::engine::Action &action : validActions) {
      const int actionIndex = ActionMap::getInstance().actionToIndex(action);
      actionMask.mutable_at(actionIndex) = 0.0;
    }

    py::object index = inferenceInstance.attr("getActionIndexForState")(observation, actionMask);
    int actionIndex = index.cast<int>();
    const sorry::engine::Action action = ActionMap::getInstance().indexToActionForPlayer(actionIndex, sorry.getPlayerTurn());
    cout << sorry.toString() << endl;
    cout << "Want to take action " << action.toString() << endl;
    sorry.doAction(action, randomEngine);
  }
}

int main() {
  // Initialize the Python interpreter
  py::scoped_interpreter guard;

  // Append the current source directory to sys.path so that we can later load any local python files. SOURCE_DIR is set from CMake.
  py::module sys = py::module::import("sys");
  const std::string sourceDir = std::string(SOURCE_DIR);
  std::cout << "Setting source directory as \"" << sourceDir << "\" (for loading python files)" << std::endl;
  sys.attr("path").cast<py::list>().append(sourceDir);

  trainReinforce();
  return 0;
}