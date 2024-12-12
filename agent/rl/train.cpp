#include "actionMap.hpp"
#include "common.hpp"
#include "actorCriticTrainingUtil.hpp"
// #include "reinforceWithBaselineTrainingUtil.hpp"
#include "trajectory.hpp"

#include <sorry/agent/random/randomAgent.hpp>
#include <sorry/agent/rl/reinforceAgent.hpp>
#include <sorry/common/common.hpp>
#include <sorry/engine/common.hpp>
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

// =================================================================================================

// Times construction to destruction and logs the time to TensorBoard.
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

class RewardTracker {
public:
  RewardTracker(const sorry::engine::Sorry &sorry, sorry::engine::PlayerColor ourColor, sorry::engine::PlayerColor opponentColor) : ourColor_(ourColor), opponentColor_(opponentColor) {
    ourLastPiecePositions_ = sorry.getPiecePositionsForPlayer(ourColor_);
    opponentLastPiecePositions_ = sorry.getPiecePositionsForPlayer(opponentColor_);
  }

  float calculateRewardForCurrentStateOfGame(const sorry::engine::Sorry &sorry) {
    auto sumAsGreen = [](const std::array<int,4> &piecePositions, sorry::engine::PlayerColor color) {
      const int count = sorry::engine::common::rotationCount(color, sorry::engine::PlayerColor::kGreen);
      int sum = 0;
      for (int pos : piecePositions) {
        sum += sorry::engine::common::rotatePosition(pos, count);
      }
      return sum;
    };

    float totalReward = 0.0;
    // Sum the positions of all pieces for each player.
    // We will scale each players' progress from [0+0+0+0, 66+66+66+66] to [0.0, 1.0].
    //  All 0's means all pieces are in start.
    //  All 66's means all pieces are in home.

    // How does our current position compare to our last position?
    const int ourLastSum = sumAsGreen(ourLastPiecePositions_, ourColor_);
    ourLastPiecePositions_ = sorry.getPiecePositionsForPlayer(ourColor_);
    const int ourCurrentSum = sumAsGreen(ourLastPiecePositions_, ourColor_);
    totalReward += (ourCurrentSum - ourLastSum) / (66.0*4);

    // How does the opponent's current position compare to their last position?
    const int opponentLastSum = sumAsGreen(opponentLastPiecePositions_, opponentColor_);
    opponentLastPiecePositions_ = sorry.getPiecePositionsForPlayer(opponentColor_);
    const int opponentCurrentSum = sumAsGreen(opponentLastPiecePositions_, opponentColor_);
    totalReward -= (opponentCurrentSum - opponentLastSum) / (66.0*4);

    return totalReward;
  }
private:
  sorry::engine::PlayerColor ourColor_;
  sorry::engine::PlayerColor opponentColor_;
  std::array<int,4> ourLastPiecePositions_;
  std::array<int,4> opponentLastPiecePositions_;
};

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
  static constexpr int kBufferSize = 101;
  std::deque<float> rewards;
  float totalReward{0.0};
};

// =================================================================================================

class Trainer {
public:
  void trainReinforce() {
    // Load the Python module
    using namespace pybind11::literals;
    py::module jaxModule = py::module::import("actor_critic");
    py::module tensorboardX = py::module::import("tensorboardX");
    summaryWriter_ = tensorboardX.attr("SummaryWriter")("flush_secs"_a=1);

    // Initialize python module/model
    // pythonTrainingUtil_ = python_wrapper::ActorCriticTrainingUtil(jaxModule, summaryWriter_, "latest");
    pythonTrainingUtil_ = python_wrapper::ActorCriticTrainingUtil(jaxModule, summaryWriter_);

    // Set up a single random agent as the first opponent for our agent
    opponentPool_.push_back(new sorry::agent::RandomAgent());
    resetOpponentStats();

    // Seed all random engines
    constexpr int kSeed = 0x533D;
    randomEngine_ = std::mt19937{kSeed};
    pythonTrainingUtil_->setSeed(kSeed);
    // Seed our opponents
    for (sorry::agent::BaseAgent *opponent : opponentPool_) {
      opponent->seed(kSeed);
    }

    // Start training
    constexpr int kEpisodeCount = 1'000'000;
    int episodeIndex = 0;
    while (episodeIndex<kEpisodeCount) {
      runEpisode(episodeIndex);
      ++episodeIndex;
    }
  }
private:
  static constexpr bool kAddSelfToPool{false};
  static constexpr int kPrintEpisodeCompletionFrequency{10};
  static constexpr int kSaveCheckpointFrequency{4};
  static constexpr sorry::engine::PlayerColor ourColor_{sorry::engine::PlayerColor::kGreen};
  static constexpr sorry::engine::PlayerColor opponentColor_{sorry::engine::PlayerColor::kBlue};
  py::object summaryWriter_;
  std::mt19937 randomEngine_;
  std::vector<sorry::agent::BaseAgent*> opponentPool_;
  std::optional<python_wrapper::ActorCriticTrainingUtil> pythonTrainingUtil_;
  std::vector<OpponentStats> opponentStats_;

  bool shouldAddSelfToPool() const {
    if (!kAddSelfToPool) {
      //  For now, we are just focused on doing as well as possible against the initial opponent.
      return false;
    }
    constexpr int kMinGamesPerOpponent = 101;
    constexpr float kMinAverageReward = 0.8; // (Reward + 1) / 2 is win rate. Average reward 0.5 is 75% win rate.
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

  void runEpisode() {
    Sorry sorry;
    RewardTracker rewardTracker;
    bool weTookAnActionBefore = false;
    std::vector<int> lastObservation;
    py::object rngKey;
    auto train = [](const std::vector<int> &observation){
      // Current observation is `sorry`.
      // Previous observation is `lastObservation`.
      // Calculate the reward for the action we took.
      float reward = rewardTracker.calculateRewardForCurrentStateOfGame(sorry);
      // We need to update the weights of the:
      //  1. Policy network, which requires:
      //    - The gradient of the log probability of the action we took
      //      - Which can come from the state and the rngkey used when selecting our action
      //    - The advantage of the action we took
      //      - advantage = reward + gamma * value(sorry) - value(sorry_before_action)
      //  2. Value network, which requires:
      //    - The state before the action was taken
      //    - The state after the action was taken
      //    - The reward we received
      pythonModel.train(lastObservation, reward, sorry, rngKey);
    };
    while (!sorry.done()) {
      if (sorry.turn() == opponent) {
        // Do opponent's turn
      } else {
        std::vector<int> observation = common::makeObservation(sorry);
        if (weTookAnActionBefore) {
          train();
        }
        // Save the current observation as the last
        lastObservation = sorry;
        // Our turn
        sorry::engine::Action action;
        std::tie(action, rngKey) = pythonModel.getActionAndRngKeyUsed(sorry);
        sorry.doAction(action);
        weTookAnActionBefore = true;
      }
    }
    // Game is done.
    train();
  }

  void runEpisode(int episodeIndex) {
    ScopedTimer timer(summaryWriter_, "entire_episode", episodeIndex);
    // Construct Sorry game
    // TODO: This construction should be moved outside the loop
    // With this initialization, we always go first, because our color is the first in the list.
    sorry::engine::Sorry sorry({ourColor_, opponentColor_});
    sorry.reset(randomEngine_);
    // Randomly choose one opponent from the pool
    std::uniform_int_distribution<size_t> dist(0, opponentPool_.size()-1);
    size_t opponentIndex = dist(randomEngine_);
    sorry::agent::BaseAgent *opponent = opponentPool_[opponentIndex];
    RewardTracker rewardTracker(sorry, ourColor_, opponentColor_);
    bool tookAnAction = false;
    float episodeTotalReward = 0.0;

    py::object logProbabilityGradient;
    std::vector<int> lastObservation;

    while (!sorry.gameDone()) {
      // Who's turn is it?
      const sorry::engine::PlayerColor playerTurn = sorry.getPlayerTurn();
      if (playerTurn == opponentColor_) {
        // It is the opponent's turn.
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
          // No need to rotate the board for an agent which is not a ReinforceAgent.
          opponent->run(sorry);
          action = opponent->pickBestAction();
        }
        sorry.doAction(action, randomEngine_);
      } else {
        // It is our turn.
        std::vector<int> observation = common::makeObservation(sorry);

        // It is our turn.
        if (tookAnAction) {
          // Calculate the reward for the previous action we took.
          float reward = rewardTracker.calculateRewardForCurrentStateOfGame(sorry);
          episodeTotalReward += reward;

          // Train
          //  Save the gradient from the last action we took (log prob)
          //  Use the current and previous observations to calculate the advantage
          pythonTrainingUtil_->train(logProbabilityGradient, lastObservation, observation);
        }

        // Take an action according to the policy, masked by the valid actions.
        const std::vector<sorry::engine::Action> actions = sorry.getActions();
        const std::vector<std::vector<int>> validActionsArray = common::createArrayOfActions(actions);
        const sorry::engine::Action action = pythonTrainingUtil_->getAction(observation, sorry.getPlayerTurn(), episodeIndex, validActionsArray);

        // Do a quick check to make sure the model's action is valid.
        if (std::find(actions.begin(), actions.end(), action) == actions.end()) {
          std::cout << "Current state: " << sorry.toString() << std::endl;
          std::cout << "Valid actions were:" << std::endl;
          for (const sorry::engine::Action &a : actions) {
            std::cout << "  " << a.toString() << std::endl;
          }
          throw std::runtime_error("Invalid action after mask "+action.toString());
        }

        // Take action in game.
        sorry.doAction(action, randomEngine_);
        tookAnAction = true;
      }
    }

    float gameResult;
    // Who won?
    sorry::engine::PlayerColor winner = sorry.getWinner();
    if (winner == ourColor_) {
      gameResult = 1.0;
    } else {
      gameResult = -1.0;
    }
    opponentStats_.at(opponentIndex).pushGameResult(gameResult);
    const float finalReward = rewardTracker.calculateRewardForCurrentStateOfGame(sorry);
    // TODO: Train
    summaryWriter_.attr("add_scalar")("episode/reward_vs_opponent_"+std::to_string(opponentIndex), episodeTotalReward, episodeIndex);
    summaryWriter_.attr("add_scalar")("episode/result_vs_opponent_"+std::to_string(opponentIndex), gameResult, episodeIndex);
    if (kAddSelfToPool) {
      summaryWriter_.attr("add_scalar")("opponent/count", opponentPool_.size(), episodeIndex);
    }

    if ((episodeIndex+1)%kPrintEpisodeCompletionFrequency == 0) {
      cout << "Episode " << episodeIndex << " complete. " << sorry::engine::toString(sorry.getWinner()) << " won" << endl;
    }
    if ((episodeIndex+1)%kSaveCheckpointFrequency == 0) {
      // pythonTrainingUtil_->saveCheckpoint();
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
