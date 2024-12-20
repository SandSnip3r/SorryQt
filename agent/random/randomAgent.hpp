#ifndef SORRY_AGENT_RANDOM_AGENT_HPP_
#define SORRY_AGENT_RANDOM_AGENT_HPP_

#include <sorry/agent/base/baseAgent.hpp>

#include <random>

namespace sorry::agent {

class RandomAgent : public BaseAgent {
public:
  RandomAgent() = default;
  RandomAgent(int seed);
  void seed(int seed) override;
  void run(const sorry::engine::Sorry &sorry) override;
  std::vector<ActionScore> getActionScores() const override;
  sorry::engine::Action pickBestAction() const override;
private:
  std::mt19937 eng_;
  sorry::engine::Action bestAction_;
  std::vector<ActionScore> actionScores_;
};

} // namespace sorry::agent

#endif // SORRY_AGENT_RANDOM_AGENT_HPP_