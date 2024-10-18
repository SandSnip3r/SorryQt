#include "sorryMcts.hpp"

#include <sorry/common/common.hpp>
#include <sorry/engine/sorry.hpp>

#include <iostream>
#include <map>

using namespace std;

// =============Board layout=============
//        30  32  34    37
// 28| |s|-|-|x| | | | |s|-|-|-|x| | |43
// 27| |                           |s|44
// 26|x|                           |.|45
// 25|.|                           |.|46
// 24|.|                           |x|47
// 23|.|                           | |48
// 22|s|                           | |49
// 21| |                           | |50
// 20| |                           | |51
// 19| |                     66|H| |s|52
// 18| |                       |.| |.|53
// 17|x|                     64|.| |.|54
// 16|.|                       |.| |.|55
// 15|.|                     62|.| |x|56
// 14|s|                   |S| |.| | |57
// 13| | |x|-|-|-|s| | | | |x|-|-|s| |58
//        11       7     4   2  60
// ======================================

class BaseAgent {
public:
  virtual sorry::engine::Action getAction(const sorry::engine::Sorry &state) = 0;
};

class RandomAgent : public BaseAgent {
public:
  RandomAgent() : eng_(sorry::common::createRandomEngine()) {}

  sorry::engine::Action getAction(const sorry::engine::Sorry &state) override {
    const auto actions = state.getActions();
    uniform_int_distribution<> dist(0, actions.size()-1);
    return actions.at(dist(eng_));
  }
private:
  std::mt19937 eng_;
};

class IterationBoundMctsAgent : public BaseAgent {
public:
  IterationBoundMctsAgent(double explorationConstant, int maxIterationCount) : mcts_(explorationConstant), maxIterationCount_(maxIterationCount) {}
  sorry::engine::Action getAction(const sorry::engine::Sorry &state) override {
    mcts_.run(state, maxIterationCount_);
    const sorry::engine::Action action = mcts_.pickBestAction();
    mcts_.reset();
    return action;
  }
private:
  SorryMcts mcts_;
  int maxIterationCount_;
};

class HumanAgent : public BaseAgent {
public:
  HumanAgent() = default;
  sorry::engine::Action getAction(const sorry::engine::Sorry &state) override {
    cout << "State: " << state.toString() << endl;
    const auto actions = state.getActions();
    for (size_t i=0; i<actions.size(); ++i) {
      cout << "  " << i << ": " << actions.at(i).toString() << endl;
    }
    int choice = -1;
    while (choice < 0 || choice >= static_cast<int>(actions.size())) {
      cout << "Please choose [0-" << actions.size()-1 << "]: ";
      cin >> choice;
    }
    return actions.at(choice);
  }
private:
};

sorry::engine::PlayerColor agentVsAgent(const std::map<sorry::engine::PlayerColor, BaseAgent*> &agents) {
  mt19937 eng = sorry::common::createRandomEngine();
  std::vector<sorry::engine::PlayerColor> playerColors;
  playerColors.reserve(agents.size());
  for (const auto &colorAndAgent : agents) {
    playerColors.push_back(colorAndAgent.first);
  }
  sorry::engine::Sorry sorry(playerColors);
  sorry.reset(eng);

  int turnNumber=0;
  while (!sorry.gameDone()) {
    // Who's turn?
    const sorry::engine::PlayerColor currentTurn = sorry.getPlayerTurn();
    BaseAgent *agent = agents.at(currentTurn);
    const sorry::engine::Action action = agent->getAction(sorry);
    sorry.doAction(action, eng);
    ++turnNumber;
  }
  return sorry.getWinner();
}

void doSingleMove() {
  mt19937 eng(123);
  SorryMcts mcts(20.0);
  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kGreen, sorry::engine::PlayerColor::kBlue});
  sorry.reset(eng);
  cout << "State " << sorry.toString() << std::endl;
  std::cout << "Actions are:" << std::endl;
  const auto actions = sorry.getActions();
  for (const auto &action : actions) {
    std::cout << "  " << action.toString() << std::endl;
  }
  mcts.run(sorry, 10000);
  sorry::engine::Action bestAction = mcts.pickBestAction();
  std::cout << "Best action is " << bestAction.toString() << endl;
}

int main() {
  HumanAgent agent1;
  RandomAgent agent2;
  std::map<sorry::engine::PlayerColor, BaseAgent*> agents = {
    {sorry::engine::PlayerColor::kGreen, &agent1},
    {sorry::engine::PlayerColor::kBlue, &agent2}
  };
  const sorry::engine::PlayerColor winner = agentVsAgent(agents);
  std::cout << toString(winner) << " won" << std::endl;
  return 0;
}