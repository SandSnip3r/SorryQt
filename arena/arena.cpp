#include <sorry/common/common.hpp>
#include <sorry/agent/rl/reinforceAgent.hpp>
#include <sorry/agent/mcts/sorryMcts.hpp>
#include <sorry/engine/sorry.hpp>

#include <pybind11/embed.h>

#include <iostream>

using namespace std;
using namespace sorry;

int main() {
  auto randomEngine = common::createRandomEngine();
  pybind11::scoped_interpreter guard;
  agent::ReinforceAgent agent1("a2c_god3");
  SorryMcts agent2(2);
  // agent::ReinforceAgent agent1("actor_critic_better_reward");
  // SorryMcts agent2(2);
  int agent1WinCount = 0;
  int agent2WinCount = 0;
  constexpr int kGameCount = 10000;
  for (int i=0; i<kGameCount; ++i) {
    engine::Sorry sorry({engine::PlayerColor::kGreen, engine::PlayerColor::kBlue});
    sorry.reset(randomEngine);
    while (!sorry.gameDone()) {
      if (sorry.getPlayerTurn() == engine::PlayerColor::kGreen) {
        agent1.run(sorry);
        // agent1.run(sorry, 1000);
        sorry::engine::Action action = agent1.pickBestAction();
        sorry.doAction(action, randomEngine);
      } else {
        // Rotate the board so that it looks like we are green.
        sorry.rotateBoard(engine::PlayerColor::kBlue, engine::PlayerColor::kGreen);
        // agent2.run(sorry);
        agent2.run(sorry, 1000);
        sorry::engine::Action action = agent2.pickBestAction();
        sorry.doAction(action, randomEngine);
        sorry.rotateBoard(engine::PlayerColor::kGreen, engine::PlayerColor::kBlue);
      }
    }
    if (sorry.getWinner() == sorry::engine::PlayerColor::kGreen) {
      ++agent1WinCount;
    } else {
      ++agent2WinCount;
    }
    cout << "Game " << i << " done. " << sorry::engine::toString(sorry.getWinner()) << " wins. " << agent1WinCount << '/' << agent2WinCount << endl;
  }

  return 0;
}