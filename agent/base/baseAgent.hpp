#ifndef SORRY_AGENT_BASE_AGENT_HPP_
#define SORRY_AGENT_BASE_AGENT_HPP_

#include <sorry/engine/action.hpp>
#include <sorry/engine/sorry.hpp>

namespace sorry::agent {

struct ActionScore {
  sorry::engine::Action action;
  double score;
  ActionScore(sorry::engine::Action action, double score) : action(action), score(score) {}
};

class BaseAgent {
public:
  std::string_view name() const { return name_; }
  void setName(std::string_view name) { name_ = name; }
  virtual void seed(int seed) = 0;
  virtual void run(const sorry::engine::Sorry &sorry) = 0;
  virtual std::vector<ActionScore> getActionScores() const = 0;
  virtual sorry::engine::Action pickBestAction() const = 0;
  virtual ~BaseAgent() = default;
private:
  std::string name_{"NO_NAME"};
};

} // namespace sorry::agent

#endif // SORRY_AGENT_BASE_AGENT_HPP_