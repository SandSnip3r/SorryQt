#include "randomAgent.hpp"

namespace sorry::agent {

RandomAgent::RandomAgent(int seed) {
  this->seed(seed);
}

void RandomAgent::seed(int seed) {
  eng_ = std::mt19937(seed);
}

void RandomAgent::run(const sorry::engine::Sorry &sorry) {
  std::vector<sorry::engine::Action> actions = sorry.getActions();
  std::uniform_int_distribution<size_t> dist(0, actions.size()-1);
  bestAction_ = actions[dist(eng_)];
  actionScores_.clear();
  actionScores_.reserve(actions.size());
  for (const sorry::engine::Action &action : actions) {
    // Since we're acting randomly, all actions are equally likely.
    actionScores_.emplace_back(action, 1.0 / actions.size());
  }
}

std::vector<ActionScore> RandomAgent::getActionScores() const {
  return actionScores_;
}

sorry::engine::Action RandomAgent::pickBestAction() const {
  return bestAction_;
}

} // namespace sorry::agent