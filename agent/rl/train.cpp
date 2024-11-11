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

namespace py = pybind11;
using namespace std;

class ScopedTimer {
public:
  ScopedTimer(py::object summaryWriter, std::string_view name, int episodeIndex) : summaryWriter_(summaryWriter), name_(name), episodeIndex_(episodeIndex) {
    startTime_ = std::chrono::high_resolution_clock::now();
  }
  ~ScopedTimer() {
    auto endTime = std::chrono::high_resolution_clock::now();
    const int microseconds = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime_).count();
    summaryWriter_.attr("add_scalar")("timing/"+name_, microseconds/1000000.0, episodeIndex_);
  }
private:
  py::object summaryWriter_;
  const std::string name_;
  const int episodeIndex_;
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime_;
};

// =================================================================================================

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

// =================================================================================================

class Trainer {
public:
  void trainReinforce() {
    // Load the Python module
    using namespace pybind11::literals;
    py::module jaxModule = py::module::import("jaxModule");
    py::module tensorboardX = py::module::import("tensorboardX");
    summaryWriter_ = tensorboardX.attr("SummaryWriter")("flush_secs"_a=1);

    // Initialize python module/model
    // pythonTrainingUtil_ = python_wrapper::TrainingUtil(jaxModule, summaryWriter_, "latest");
    pythonTrainingUtil_ = python_wrapper::TrainingUtil(jaxModule, summaryWriter_);

    // Seed all random engines
    constexpr int kSeed = 0x5EED;
    randomEngine_ = std::mt19937{kSeed};
    pythonTrainingUtil_->setSeed(kSeed);

    // Start training
    constexpr int kEpisodeCount = 1'000'000;
    int episodeIndex = 0;
    constexpr int kBatchSize = 1;
    std::vector<Trajectory> batchTrajectories(kBatchSize);
    while (episodeIndex<kEpisodeCount) {
      for (int i=0; i<kBatchSize; ++i) {
        if (episodeIndex >= kEpisodeCount) {
          break;
        }
        collectTrajectory(episodeIndex, batchTrajectories[i]);
        ++episodeIndex;
      }
      // A batch of trajectories has been generated, now train the model
      {
        ScopedTimer timer(summaryWriter_, "training_util_train", episodeIndex);
        pythonTrainingUtil_->train(batchTrajectories, episodeIndex);
      }
    }
  }
private:
  py::object summaryWriter_;
  std::mt19937 randomEngine_;
  std::optional<python_wrapper::TrainingUtil> pythonTrainingUtil_;

  void collectTrajectory(int episodeIndex, Trajectory &trajectory) {
    ScopedTimer timer(summaryWriter_, "entire_episode", episodeIndex);
    // Construct Sorry game
    // TODO: This construction should be moved outside the loop
    const sorry::engine::PlayerColor playerColor = pickRandomColor();
    sorry::engine::Sorry sorry({playerColor});
    sorry.reset(randomEngine_);
    trajectory.reset();

    // Generate a full trajectory according to the policy
    int actionCount = 0;
    while (!sorry.gameDone()) {
      const std::vector<sorry::engine::Action> actions = sorry.getActions();
      py::object policyGradient;
      sorry::engine::Action action;
      py::object observation = common::makeNumpyObservation(sorry);
      // TODO: Rather than getting the value & value gradient here, I could simply save the observation in the trajectory and get it later during training.
      py::object valueGradient;
      float value;
      std::tie(valueGradient, value) = pythonTrainingUtil_->getValueGradientAndValue(observation);

      // Take an action according to the policy, masked by the valid actions
      std::tie(policyGradient, action) = pythonTrainingUtil_->getPolicyGradientAndAction(observation, sorry.getPlayerTurn(), episodeIndex, actions);
      // std::cout << "Taking action " << action.toString() << std::endl;

      if (std::find(actions.begin(), actions.end(), action) == actions.end()) {
        std::cout << "Current state: " << sorry.toString() << std::endl;
        std::cout << "Valid actions were:" << std::endl;
        for (const sorry::engine::Action &a : actions) {
          std::cout << "  " << a.toString() << std::endl;
        }
        throw std::runtime_error("Invalid action after mask "+action.toString());
      }

      // Take action in game
      sorry.doAction(action, randomEngine_);
      ++actionCount;

      float reward;
      if (sorry.gameDone()) {
        reward = 0.0;
      } else {
        // Give a small negative reward to encourage the model to finish the game as quickly as possible
        reward = -0.01;
      }

      // Store the observation into a python-read trajectory data structure
      trajectory.pushStep(policyGradient, reward, valueGradient, value);
    }
    summaryWriter_.attr("add_scalar")("episode/action_count", actionCount, episodeIndex);

    if ((episodeIndex+1)%10 == 0) {
      cout << "Episode " << episodeIndex << " complete" << endl;
      if ((episodeIndex+1)%10 == 0) {
        pythonTrainingUtil_->saveCheckpoint();
      }
    }
  }

  sorry::engine::PlayerColor pickRandomColor() {
    const std::array<sorry::engine::PlayerColor, 4> colors = {
      sorry::engine::PlayerColor::kGreen,
      sorry::engine::PlayerColor::kYellow,
      sorry::engine::PlayerColor::kRed,
      sorry::engine::PlayerColor::kBlue
    };
    std::uniform_int_distribution<size_t> dist(0, colors.size()-1);
    return colors[dist(randomEngine_)];
  };
};

// =================================================================================================

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

// =================================================================================================

int main() {
  // Initialize the Python interpreter
  py::scoped_interpreter guard;

  // Append the current source directory to sys.path so that we can later load any local python files. SOURCE_DIR is set from CMake.
  py::module sys = py::module::import("sys");
  const std::string sourceDir = std::string(SOURCE_DIR);
  std::cout << "Setting source directory as \"" << sourceDir << "\" (for loading python files)" << std::endl;
  sys.attr("path").cast<py::list>().append(sourceDir);

  Trainer trainer;
  trainer.trainReinforce();
  return 0;
}