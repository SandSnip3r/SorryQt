#include "actionMap.hpp"
#include "common.hpp"
#include "trainingUtil.hpp"
#include "trajectory.hpp"

#include <sorry/agent/random/randomAgent.hpp>
#include <sorry/agent/rl/reinforceAgent.hpp>
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
#include <deque>
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
  constexpr int kSeed = 123;
  mt19937 randomEngine(kSeed);
  sorry::agent::RandomAgent randomAgent;
  randomAgent.seed(kSeed);
  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kGreen});
  for (int i=0; i<kNumGames; ++i) {
    int thisGameActionCount = 0;
    sorry.reset(randomEngine);
    while (!sorry.gameDone()) {
      randomAgent.run(sorry);
      const sorry::engine::Action action = randomAgent.pickBestAction();
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

struct OpponentStats {
public:
  void pushGameResult(float reward) {
    if (rewards.size() >= kBufferSize) {
      totalReward -= rewards.front();
      rewards.pop_front();
    }
    rewards.push_back(reward);
    totalReward += reward;
  }

  int gameCount() const {
    return rewards.size();
  }

  float averageReward() const {
    if (rewards.empty()) {
      return 0.0;
    }
    return totalReward / rewards.size();
  }
private:
  static constexpr int kBufferSize = 31;
  std::deque<float> rewards;
  float totalReward{0.0};
};

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

    // Set up a single random agent as the first opponent for our agent
    opponentPool_.push_back(new sorry::agent::RandomAgent());
    resetOpponentStats();

    // Seed all random engines
    constexpr int kSeed = 0x5EED;
    randomEngine_ = std::mt19937{kSeed};
    pythonTrainingUtil_->setSeed(kSeed);
    // Also see our random opponent
    opponentPool_.back()->seed(kSeed);

    // Start training
    constexpr int kEpisodeCount = 1'000'000;
    int episodeIndex = 0;
    constexpr int kBatchSize = 1;
    std::vector<Trajectory> batchTrajectories(kBatchSize);
    while (episodeIndex<kEpisodeCount) {
      for (int i=0; i<kBatchSize; ++i) {
        if (episodeIndex >= kEpisodeCount) {
          // TODO: If this ends on an incomplete batch, the model should not be trained.
          break;
        }
        collectTrajectory(episodeIndex, batchTrajectories[i]);
        ++episodeIndex;
      }
      if (shouldAddSelfToPool()) {
        std::cout << "Going to add self to pool" << std::endl;
        opponentPool_.push_back(new sorry::agent::ReinforceAgent(pythonTrainingUtil_->getPythonTrainingUtilInstance()));
        opponentPool_.back()->seed(kSeed);
        resetOpponentStats();
      }
      // A batch of trajectories has been generated, now train the model
      {
        ScopedTimer timer(summaryWriter_, "training_util_train", episodeIndex);
        pythonTrainingUtil_->train(batchTrajectories, episodeIndex);
      }
    }
  }
private:
  static constexpr sorry::engine::PlayerColor ourColor_{sorry::engine::PlayerColor::kGreen};
  static constexpr sorry::engine::PlayerColor opponentColor_{sorry::engine::PlayerColor::kBlue};
  py::object summaryWriter_;
  std::mt19937 randomEngine_;
  std::vector<sorry::agent::BaseAgent*> opponentPool_;
  std::optional<python_wrapper::TrainingUtil> pythonTrainingUtil_;
  std::vector<OpponentStats> opponentStats_;

  bool shouldAddSelfToPool() const {
    constexpr int kMinGamesPerOpponent = 31;
    constexpr float kMinAverageReward = 0.5; // (Reward + 1) / 2 is win rate. Average reward 0.5 is 75% win rate.
    // We should have played against every opponent at least `kMinGamesPerOpponent` times, for statistical significance.
    // The minimum average reward should be at least `kMinAverageReward`.
    std::cout << "Stats: Game counts: [ ";
    for (const OpponentStats &opponentStats : opponentStats_) {
      std::cout << opponentStats.gameCount() << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "    Average reward: [ ";
    for (const OpponentStats &opponentStats : opponentStats_) {
      std::cout << opponentStats.averageReward() << ", ";
    }
    std::cout << "]" << std::endl;
    for (const OpponentStats &opponentStats : opponentStats_) {
      if (opponentStats.gameCount() < kMinGamesPerOpponent) {
        return false;
      }
    }
    auto minElementIt = std::min_element(opponentStats_.begin(), opponentStats_.end(), [](const OpponentStats &a, const OpponentStats &b) {
      return a.averageReward() < b.averageReward();
    });
    return minElementIt->averageReward() >= kMinAverageReward;
  }

  void resetOpponentStats() {
    opponentStats_.clear();
    opponentStats_.resize(opponentPool_.size());
  }

  void collectTrajectory(int episodeIndex, Trajectory &trajectory) {
    ScopedTimer timer(summaryWriter_, "entire_episode", episodeIndex);
    // Construct Sorry game
    // TODO: This construction should be moved outside the loop
    // With this initialization, we always go first, because our color is the first in the list.
    sorry::engine::Sorry sorry({ourColor_, opponentColor_});
    sorry.reset(randomEngine_);
    trajectory.reset();
    // Randomly choose one opponent from the pool
    std::uniform_int_distribution<size_t> dist(0, opponentPool_.size()-1);
    size_t opponentIndex = dist(randomEngine_);
    std::cout << "Playing against opponent #" << opponentIndex << std::endl;
    sorry::agent::BaseAgent *opponent = opponentPool_[opponentIndex];

    // Generate a full trajectory according to the policy
    while (!sorry.gameDone()) {
      // Who's turn is it?
      const sorry::engine::PlayerColor playerTurn = sorry.getPlayerTurn();
      if (playerTurn == opponentColor_) {
        // Let the opponent take a turn
        sorry::engine::Action action;
        if (dynamic_cast<sorry::agent::ReinforceAgent*>(opponent) != nullptr) {
          // Reinforce Agent was trained to play as green. Rotate the board so that they play from our position.
          sorry.rotateBoard(opponentColor_, ourColor_);
          opponent->run(sorry);
          action = opponent->pickBestAction();
          action.rotateBoard(ourColor_, opponentColor_);
          // Put board back.
          sorry.rotateBoard(ourColor_, opponentColor_);
        } else {
          opponent->run(sorry);
          action = opponent->pickBestAction();
        }
        sorry.doAction(action, randomEngine_);
      } else {
        // It is our turn
        py::object policyGradient;
        py::object valueGradient;
        float value;
        py::object observation = common::makeNumpyObservation(sorry);

        // TODO: Rather than getting the value & value gradient here, I could simply save the observation in the trajectory and get it later during training.
        std::tie(valueGradient, value) = pythonTrainingUtil_->getValueGradientAndValue(observation);

        // Take an action according to the policy, masked by the valid actions
        const std::vector<sorry::engine::Action> actions = sorry.getActions();
        sorry::engine::Action action;
        std::tie(policyGradient, action) = pythonTrainingUtil_->getPolicyGradientAndAction(observation, sorry.getPlayerTurn(), episodeIndex, actions);

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

        // Store the observation into a python-read trajectory data structure
        trajectory.pushStep(policyGradient, 0.0, valueGradient, value);
      }
    }

    float reward;
    // Who won?
    sorry::engine::PlayerColor winner = sorry.getWinner();
    if (winner == ourColor_) {
      reward = 1.0;
    } else {
      reward = -1.0;
    }
    opponentStats_.at(opponentIndex).pushGameResult(reward);
    trajectory.setLastReward(reward);
    summaryWriter_.attr("add_scalar")("episode/reward_vs_opponent_"+std::to_string(opponentIndex), reward, episodeIndex);
    summaryWriter_.attr("add_scalar")("opponent/count", opponentPool_.size(), episodeIndex);

    if ((episodeIndex+1)%100 == 0) {
      cout << "Episode " << episodeIndex << " complete. " << sorry::engine::toString(sorry.getWinner()) << " won" << endl;
      if ((episodeIndex+1)%1000 == 0) {
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
