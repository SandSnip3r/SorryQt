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
  OpponentStats(int maxBufferSize) : maxBufferSize(maxBufferSize) {}

  void pushGameResult(float result) {
    if (results.size() >= maxBufferSize) {
      resultSum -= results.front();
      results.pop_front();
    }
    results.push_back(result);
    resultSum += result;
  }

  int gameCount() const {
    return results.size();
  }

  float averageResult() const {
    if (results.empty()) {
      return 0.0;
    }
    return resultSum / results.size();
  }
private:
  const int maxBufferSize{1};
  std::deque<float> results;
  float resultSum{0.0};
};

// =================================================================================================

std::pair<float, float> calculateMeanAndStdDev(const std::vector<float> &data) {
  float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();

  // Use std::transform to compute squared differences
  float variance = std::transform_reduce(
      data.begin(), data.end(), 0.0f, std::plus<>(),
      [mean](float value) { return (value - mean) * (value - mean); }
  ) / data.size();

  return {mean, std::sqrt(variance)};
}

char colorToChar(sorry::engine::PlayerColor color) {
  return sorry::engine::toString(color)[0];
}

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
    pythonTrainingUtil_ = python_wrapper::ActorCriticTrainingUtil(jaxModule, summaryWriter_, kRestoreFromCheckpoint);

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

      if (shouldAddSelfToPool()) {
        cout << "Adding self to pool" << endl;
        opponentPool_.push_back(new sorry::agent::ReinforceAgent(pythonTrainingUtil_->getPythonTrainingUtilInstance()));
        opponentPool_.back()->seed(kSeed);
        resetOpponentStats();
      }
      ++episodeIndex;
    }
  }
private:
<<<<<<< HEAD
  static constexpr bool kAddSelfToPool{false};
  static constexpr int kPrintEpisodeCompletionFrequency{10};
  static constexpr int kSaveCheckpointFrequency{1000};
=======
  static constexpr bool kRestoreFromCheckpoint{false};
  static constexpr bool kAddSelfToPool{true};
  static constexpr int kPrintEpisodeCompletionFrequency{20};
  static constexpr int kSaveCheckpointFrequency{1000};
  static constexpr int kMinGamesPerOpponent{250};
>>>>>>> actor-critic
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
    constexpr float kMinAverageResult = 0.4;
    // We should have played against every opponent at least `kMinGamesPerOpponent` times, for statistical significance.
    // The minimum average result should be at least `kMinAverageResult`.
    std::cout << "Stats: [ ";
    for (const OpponentStats &opponentStats : opponentStats_) {
      std::cout << "{" << opponentStats.gameCount() << ", " << opponentStats.averageResult() << "}, ";
    }
    std::cout << "]" << std::endl;
    for (const OpponentStats &opponentStats : opponentStats_) {
      if (opponentStats.gameCount() < kMinGamesPerOpponent) {
        return false;
      }
    }
    auto minElementIt = std::min_element(opponentStats_.begin(), opponentStats_.end(), [](const OpponentStats &a, const OpponentStats &b) {
      return a.averageResult() < b.averageResult();
    });
    return minElementIt->averageResult() >= kMinAverageResult;
  }

  void resetOpponentStats() {
    opponentStats_.clear();
    for (int i=0; i<opponentPool_.size(); ++i) {
      opponentStats_.emplace_back(kMinGamesPerOpponent);
    }
  }

  // Throws if action is invalid.
  void checkIfActionIsValid(const sorry::engine::Sorry &sorry, const std::vector<sorry::engine::Action> &validActions, const sorry::engine::Action &action) {
    if (std::find(validActions.begin(), validActions.end(), action) == validActions.end()) {
      std::cout << "Current state: " << sorry.toString() << std::endl;
      std::cout << "Valid actions were:" << std::endl;
      for (const sorry::engine::Action &a : validActions) {
        std::cout << "  " << a.toString() << std::endl;
      }
      throw std::runtime_error("Invalid action after mask "+action.toString());
    }
  }

  void runEpisode(int episodeIndex) {
    ScopedTimer timer(summaryWriter_, "entire_episode", episodeIndex);

    // Reset the game environment.
    sorry::engine::Sorry sorry({ourColor_, opponentColor_});
    sorry.reset(randomEngine_);

    // Randomly choose one opponent from the pool.
    std::uniform_int_distribution<size_t> dist(0, opponentPool_.size()-1);
    size_t opponentIndex = dist(randomEngine_);
    sorry::agent::BaseAgent *opponent = opponentPool_[opponentIndex];

    // Use a separate structure for determining rewards.
    RewardTracker rewardTracker(sorry, ourColor_, opponentColor_);
    float episodeTotalReward = 0.0;

    // Set up some data structures used during the episode.
    std::vector<int> currentObservation = common::makeObservation(sorry);
    std::vector<int> lastObservation;
    std::vector<std::vector<int>> lastValidActionsArray;
    py::object rngKey;
    std::vector<float> valueFunctionLosses;

    // Define the training function.
    auto train = [&](){
      // Current observation is `sorry`.
      // Previous observation is `lastObservation`.
      // Calculate the reward for the action we took.
      float reward = rewardTracker.calculateRewardForCurrentStateOfGame(sorry);
      episodeTotalReward += reward;
      // We need to update the weights of the:
      //  1. Policy network, which requires:
      //    - The gradient of the log probability of the action we took
      //      - Which can come from the state and the rngkey used when selecting our action
      //    - The advantage of the action we took
      //      - advantage = reward + gamma * value(currentObservation) - value(lastObservation)
      //  2. Value network, which requires:
      //    - The state before the action was taken
      //    - The state after the action was taken
      //    - The reward we received
      float valueFunctionLoss = pythonTrainingUtil_->train(lastObservation, reward, currentObservation, rngKey, lastValidActionsArray);
      valueFunctionLosses.push_back(valueFunctionLoss);
    };

    // Run an entire episode.
    bool weTookAnActionBefore = false;
    int stepIndex=0;
    while (!sorry.gameDone()) {
      const sorry::engine::PlayerColor playerTurn = sorry.getPlayerTurn();
      if (playerTurn == opponentColor_) {
        // Do opponent's turn
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
        // Our turn
        currentObservation = common::makeObservation(sorry);
        if (weTookAnActionBefore) {
          train();
        }
        // Take an action according to the policy, masked by the valid actions.
        const std::vector<sorry::engine::Action> validActions = sorry.getActions();
        const std::vector<std::vector<int>> validActionsArray = common::createArrayOfActions(validActions);
        sorry::engine::Action action;
        std::tie(action, rngKey) = pythonTrainingUtil_->getActionAndKeyUsed(currentObservation, playerTurn, episodeIndex, validActionsArray);

        // Do a quick check to make sure the model's action is valid.
        checkIfActionIsValid(sorry, validActions, action);

        sorry.doAction(action, randomEngine_);
        weTookAnActionBefore = true;
        // Save the current observation as the last
        lastObservation = currentObservation;
        lastValidActionsArray = validActionsArray;
      }
      ++stepIndex;
    }

    // Game is done.
    train();

    // Log reward.
    summaryWriter_.attr("add_scalar")("total_reward_vs_opponent_"+std::to_string(opponentIndex), episodeTotalReward, episodeIndex);

    // Log the game result.
    float gameResult;
    sorry::engine::PlayerColor winner = sorry.getWinner();
    if (winner == ourColor_) {
      gameResult = 1.0;
    } else {
      gameResult = -1.0;
    }
    summaryWriter_.attr("add_scalar")("result_vs_opponent_"+std::to_string(opponentIndex), gameResult, episodeIndex);

    // Track the our win rate against the opponent.
    opponentStats_.at(opponentIndex).pushGameResult(gameResult);

    // Log stats about the losses.
    auto [valueLossMean, valueLossStdDev] = calculateMeanAndStdDev(valueFunctionLosses);
    summaryWriter_.attr("add_scalar")("value_loss/mean", valueLossMean, episodeIndex);
    summaryWriter_.attr("add_scalar")("value_loss/std_dev", valueLossStdDev, episodeIndex);

    // Log stats about our opponents.
    if (kAddSelfToPool) {
      summaryWriter_.attr("add_scalar")("opponent_count", opponentPool_.size(), episodeIndex);
    }

    if ((episodeIndex+1)%kPrintEpisodeCompletionFrequency == 0) {
      cout << episodeIndex+1 << " episodes completed" << endl;
    }
    if ((episodeIndex+1)%kSaveCheckpointFrequency == 0) {
      pythonTrainingUtil_->saveCheckpoint();
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
