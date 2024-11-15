#ifndef SORRY_AGENT_REINFORCE_AGENT_HPP_
#define SORRY_AGENT_REINFORCE_AGENT_HPP_

#include <sorry/agent/base/baseAgent.hpp>
#include <sorry/engine/action.hpp>
#include <sorry/engine/sorry.hpp>

#include <pybind11/pybind11.h>

#include <optional>

namespace sorry::agent {

class __attribute__ ((visibility("hidden"))) ReinforceAgent : public sorry::agent::BaseAgent {
public:
  ReinforceAgent();
  ReinforceAgent(pybind11::object trainingUtilInstance);
  void seed(int seed) override;

  // After calling run, action preferences remain until the next call to run.
  void run(const sorry::engine::Sorry &sorry) override;
  std::vector<ActionScore> getActionScores() const override;
  sorry::engine::Action pickBestAction() const override;
private:
  pybind11::module jaxModule_;
  pybind11::object inferenceInstance_;
  pybind11::object jaxRandomKey_;
  std::optional<sorry::engine::Action> bestAction_;
  std::vector<ActionScore> actionScores_;

  void initializeJaxModule();
};

} // namespace sorry::agent

#endif // SORRY_AGENT_REINFORCE_AGENT_HPP_