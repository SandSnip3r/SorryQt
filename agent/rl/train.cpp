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

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
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
    reset(sorry);
  }

  void reset(const sorry::engine::Sorry &sorry) {
    ourLastSum_ = sumAsGreen(sorry.getPiecePositionsForPlayer(ourColor_), ourColor_);
    opponentLastSum_ = sumAsGreen(sorry.getPiecePositionsForPlayer(opponentColor_), opponentColor_);
  }

  float calculateRewardForCurrentStateOfGame(const sorry::engine::Sorry &sorry, int stepIndex, py::object &summaryWriter) {
    const float ratio = static_cast<float>(std::min(stepIndex, kStepsToAnnealOver)) / kStepsToAnnealOver;
    if ((stepIndex+1) % 1000 == 0) {
      summaryWriter.attr("add_scalar")("reward_ratio", ratio, stepIndex);
    }
    return calculatePerfectReward(sorry) * ratio + calculateDenseReward(sorry) * (1.0 - ratio);
  }
private:
  static constexpr int kStepsToAnnealOver{1'000'000};
  sorry::engine::PlayerColor ourColor_;
  sorry::engine::PlayerColor opponentColor_;
  int ourLastSum_;
  int opponentLastSum_;

  int sumAsGreen(const std::array<int,4> &piecePositions, sorry::engine::PlayerColor color) {
    const int count = sorry::engine::common::rotationCount(color, sorry::engine::PlayerColor::kGreen);
    int sum = 0;
    for (int pos : piecePositions) {
      sum += sorry::engine::common::rotatePosition(pos, count);
    }
    return sum;
  };

  float calculatePerfectReward(const sorry::engine::Sorry &sorry) const {
    if (!sorry.gameDone()) {
      return 0.0;
    }
    if (sorry.getWinner() == ourColor_) {
      return 1.0;
    }
    return -1.0;
  }

  float calculateDenseReward(const sorry::engine::Sorry &sorry) {
    float totalReward = 0.0;
    // Sum the positions of all pieces for each player.
    // We will scale each players' progress from [0+0+0+0, 66+66+66+66] to [0.0, 1.0].
    //  All 0's means all pieces are in start.
    //  All 66's means all pieces are in home.

    // How does our current position compare to our last position?
    const int ourCurrentSum = sumAsGreen(sorry.getPiecePositionsForPlayer(ourColor_), ourColor_);
    totalReward += (ourCurrentSum - ourLastSum_) / (66.0*4);
    ourLastSum_ = ourCurrentSum;

    // How does the opponent's current position compare to their last position?
    const int opponentCurrentSum = sumAsGreen(sorry.getPiecePositionsForPlayer(opponentColor_), opponentColor_);
    totalReward -= (opponentCurrentSum - opponentLastSum_) / (66.0*4);
    opponentLastSum_ = opponentCurrentSum;

    return totalReward;
  }
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

// =================================================================================================

class OpponentPool {
public:
  void addOpponent(sorry::agent::BaseAgent *agent) {
    opponents_.push_back(agent);
  }
  sorry::agent::BaseAgent* pickRandomOpponent(std::mt19937 &randomEngine) {
    std::vector<int> weights(opponents_.size());
    std::iota(weights.begin(), weights.end(), 2);
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    // std::uniform_int_distribution<size_t> dist(0, opponents_.size()-1);
    return opponents_[dist(randomEngine)];
  }
  int size() const {
    return opponents_.size();
  }
  const std::vector<sorry::agent::BaseAgent*>& opponents() const {
    return opponents_;
  }
private:
  std::vector<sorry::agent::BaseAgent*> opponents_;
};

// =================================================================================================

// An environment consists of:
//  - A Sorry engine
//  - An opponent
//  - A reward tracker
// TODO: For now, we assume that the trained agent always goes first. If instead the "opponent" went first, we would need to have them take their action before returning an observation.
class Environment {
public:
  void setOpponent(sorry::agent::BaseAgent *opponent) {
    opponent_ = opponent;
  }

  const sorry::agent::BaseAgent *getOpponent() const {
    return opponent_;
  }

  std::pair<float, std::optional<sorry::engine::PlayerColor>> step(const sorry::engine::Action &action, std::mt19937 &randomEngine, int stepIndex, OpponentPool &opponentPool, py::object summaryWriter) {
    if (sorry_.getPlayerTurn() != kOurColor_) {
      throw std::runtime_error("It is not our turn to play.");
    }
    sorry_.doAction(action, randomEngine);
    const float reward = rewardTracker_.calculateRewardForCurrentStateOfGame(sorry_, stepIndex, summaryWriter);

    // Let the opponent take their turn.
    while (!sorry_.gameDone() && sorry_.getPlayerTurn() == kOpponentColor_) {
      if (dynamic_cast<sorry::agent::ReinforceAgent*>(opponent_) != nullptr) {
        // Rotate board since the ReinforceAgent only knows how to play as green.
        sorry_.rotateBoard(kOpponentColor_, kOurColor_);
        opponent_->run(sorry_);
        sorry_.doAction(opponent_->pickBestAction(), randomEngine);
        sorry_.rotateBoard(kOurColor_, kOpponentColor_);
      } else {
        // Opponent is not restricted to playing as green.
        opponent_->run(sorry_);
        sorry_.doAction(opponent_->pickBestAction(), randomEngine);
      }
    }

    std::optional<sorry::engine::PlayerColor> winner;
    if (sorry_.gameDone()) {
      winner = sorry_.getWinner();
    }

    if (winner.has_value()) {
      reset(randomEngine, opponentPool);
    }

    return {reward, winner};
  }

  void reset(std::mt19937 &randomEngine, OpponentPool &opponentPool) {
    sorry_.reset(randomEngine);
    rewardTracker_.reset(sorry_);
    // Grab a new opponent for the duration of the next game.
    cout << "Updating opponent from ";
    if (opponent_ != nullptr) {
      cout << opponent_->name();
    } else {
      cout << "NONE";
    }
    setOpponent(opponentPool.pickRandomOpponent(randomEngine));
    cout << " to " << opponent_->name() << endl;
  }

  const sorry::engine::Sorry& getSorry() const {
    return sorry_;
  }

  bool done() const {
    return sorry_.gameDone();
  }
private:
  const sorry::engine::PlayerColor kOurColor_{sorry::engine::PlayerColor::kGreen};
  const sorry::engine::PlayerColor kOpponentColor_{sorry::engine::PlayerColor::kBlue};
  sorry::engine::Sorry sorry_{ kOurColor_, kOpponentColor_ };
  RewardTracker rewardTracker_{sorry_, kOurColor_, kOpponentColor_};
  sorry::agent::BaseAgent *opponent_{nullptr};
};

// =================================================================================================

template<int EnvironmentCount>
class TrainingData {
public:
  void saveObservation(int environmentIndex, std::vector<int> &&observation) {
    previousObservations_[environmentIndex] = nextObservations_[environmentIndex];
    nextObservations_[environmentIndex] = std::move(observation);
  }

  void saveActionInfo(int environmentIndex, const py::object &rngKey, std::vector<std::vector<int>> &&validActionsArray) {
    rngKeys_[environmentIndex] = rngKey;
    validActionsArrays_[environmentIndex] = std::move(validActionsArray);
  }

  void saveReward(int environmentIndex, float reward) {
    rewards_[environmentIndex] = reward;
  }

  void saveGameDone(int environmentIndex, bool gameDone) {
    gameDones_[environmentIndex] = gameDone;
  }

  const std::array<std::vector<int>, EnvironmentCount>& getPreviousObservations() const {
    return previousObservations_;
  }

  const std::array<std::vector<int>, EnvironmentCount>& getNextObservations() const {
    return nextObservations_;
  }

  const std::array<py::object, EnvironmentCount>& getRngKeys() const {
    return rngKeys_;
  }

  const std::array<std::vector<std::vector<int>>, EnvironmentCount>& getValidActionsArrays() const {
    return validActionsArrays_;
  }

  const std::array<float, EnvironmentCount>& getRewards() const {
    return rewards_;
  }

  const std::array<bool, EnvironmentCount>& getGameDones() const {
    return gameDones_;
  }
private:
  std::array<std::vector<int>, EnvironmentCount> previousObservations_;
  std::array<std::vector<int>, EnvironmentCount> nextObservations_;
  std::array<py::object, EnvironmentCount> rngKeys_;
  std::array<std::vector<std::vector<int>>, EnvironmentCount> validActionsArrays_;
  std::array<float, EnvironmentCount> rewards_;
  std::array<bool, EnvironmentCount> gameDones_;
};

// =================================================================================================

class Trainer {
public:
  void trainReinforce() {
    using namespace pybind11::literals;
    py::module jaxModule = py::module::import("actor_critic");
    py::module tensorboardX = py::module::import("tensorboardX");
    summaryWriter_ = tensorboardX.attr("SummaryWriter")("flush_secs"_a=1);
    python_wrapper::ActorCriticTrainingUtil pythonTrainingUtil(jaxModule, summaryWriter_, kRestoreFromCheckpoint);

    constexpr int kEnvironmentCount = 128;
    constexpr int kSeed = 0x533D;
    mt19937 randomEngine{kSeed};
    pythonTrainingUtil.setSeed(kSeed);

    // Add an agent which acts randomly as our first opponent.
    sorry::agent::BaseAgent *randomAgent = new sorry::agent::RandomAgent();
    randomAgent->seed(kSeed);
    randomAgent->setName("RandomAgent");
    opponentPool_.addOpponent(randomAgent);
    opponentStats_.emplace(std::piecewise_construct,
                            std::forward_as_tuple(opponentPool_.opponents().back()),
                            std::forward_as_tuple(kMinGamesPerOpponent));

    // Create & reset all environments
    array<Environment, kEnvironmentCount> environments;
    for (Environment &environment : environments) {
      environment.reset(randomEngine, opponentPool_);
    }

    int stepIndex = 0;
    while (true) {
      // Take a step in each environment, meanwhile collecting data for training.
      for (int environmentIndex=0; environmentIndex<environments.size(); ++environmentIndex) {
        Environment &environment = environments[environmentIndex];
        // Take a step in the environment.
        const sorry::engine::Sorry &sorry = environment.getSorry();
        if (sorry.getPlayerTurn() != sorry::engine::PlayerColor::kGreen) {
          throw std::runtime_error("trainReinforce(): It is not our turn to play.");
        }
        std::vector<int> currentObservation = common::makeObservation(sorry);

        // Take an action according to the policy, masked by the valid actions.
        const std::vector<sorry::engine::Action> validActions = sorry.getActions();
        std::vector<std::vector<int>> validActionsArray = common::createArrayOfActions(validActions);
        sorry::engine::Action action;
        py::object rngKey;
        std::tie(action, rngKey) = pythonTrainingUtil.getActionAndKeyUsed(currentObservation, sorry::engine::PlayerColor::kGreen, stepIndex, validActionsArray);

        // Do a quick check to make sure the model's action is valid.
        checkIfActionIsValid(sorry, validActions, action);

        auto [reward, winner] = environment.step(action, randomEngine, stepIndex, opponentPool_, summaryWriter_);

        if (winner.has_value()) {
          ++completedEpisodeCount_;
          cout << "Episode #" << completedEpisodeCount_ << " completed (Environment #" << environmentIndex << ")" << endl;
          // Save the result of the game.
          const float gameResult = (*winner == sorry::engine::PlayerColor::kGreen ? 1.0 : -1.0);
          addResultForOpponent(gameResult, environment.getOpponent(), stepIndex);

          // Maybe add self to pool.
          if (shouldAddSelfToPool()) {
            cout << "Adding self to pool" << endl;
            sorry::agent::BaseAgent *selfAsOpponent = new sorry::agent::ReinforceAgent(pythonTrainingUtil.getPythonTrainingUtilInstance());
            selfAsOpponent->seed(kSeed);
            selfAsOpponent->setName("self_v" + std::to_string(opponentPool_.size()-1));
            opponentPool_.addOpponent(selfAsOpponent);
            opponentStats_.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(opponentPool_.opponents().back()),
                                    std::forward_as_tuple(kMinGamesPerOpponent));
          }
        }

        const std::vector<int> nextObservation = common::makeObservation(environment.getSorry());
        float valueFunctionLoss = pythonTrainingUtil.train(currentObservation, reward, nextObservation, rngKey, validActionsArray, winner.has_value());
        summaryWriter_.attr("add_scalar")("value_loss", valueFunctionLoss, stepIndex);
        summaryWriter_.attr("add_scalar")("opponent_count", opponentPool_.size(), stepIndex);

        // Update the step index.
        ++stepIndex;
      }
      cout << stepIndex << " steps completed" << endl;
    }
  }
private:
  static constexpr bool kRestoreFromCheckpoint{false};
  static constexpr bool kAddSelfToPool{true};
  static constexpr int kPrintEpisodeCompletionFrequency{20};
  static constexpr int kSaveCheckpointFrequency{1000};
  static constexpr int kMinGamesPerOpponent{256};
  int completedEpisodeCount_{0};

  // A pool of opponents to train against.
  OpponentPool opponentPool_;
  py::object summaryWriter_;
  std::map<const sorry::agent::BaseAgent*,OpponentStats> opponentStats_;

  bool shouldAddSelfToPool() const {
    if (!kAddSelfToPool) {
      //  For now, we are just focused on doing as well as possible against the initial opponent.
      return false;
    }
    constexpr float kMinAverageResult = 0.2;
    // We should have played against every opponent at least `kMinGamesPerOpponent` times, for statistical significance.
    // The minimum average result should be at least `kMinAverageResult`.
    std::cout << "Stats: [\n";
    for (const auto &[opponent, opponentStats] : opponentStats_) {
      std::cout << "  { " << opponent->name() << ", " << opponentStats.gameCount() << ", " << opponentStats.averageResult() << " },\n";
    }
    std::cout << "]" << std::endl;
    for (const auto &[opponent, opponentStats] : opponentStats_) {
      if (opponentStats.gameCount() < kMinGamesPerOpponent) {
        return false;
      }
    }
    auto minElementIt = std::min_element(opponentStats_.begin(), opponentStats_.end(), [](const auto &lhs, const auto &rhs) {
      return lhs.second.averageResult() < rhs.second.averageResult();
    });
    return minElementIt->second.averageResult() >= kMinAverageResult;
  }

  void addResultForOpponent(float result, const sorry::agent::BaseAgent *opponent, int stepIndex) {
    opponentStats_.at(opponent).pushGameResult(result);
    summaryWriter_.attr("add_scalar")("result_vs_"+std::string(opponent->name()), result, stepIndex);
  }
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
