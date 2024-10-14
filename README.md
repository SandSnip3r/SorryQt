# Sorry

The repository is for the board game Sorry.

![sorry_board](engine/images/board.png)

## Contents

- [agent/mcts/](agent/mcts/)
  - An implementation of Monte Carlo Tree Search for Sorry
- [agent/rl/](agent/rl/)
  - Reinforcement Learning for Sorry
- [engine/](engine/)
  - The core logic of the game Sorry
- [ui/](ui/)
  - A Qt-based UI for playing Sorry against various agents

## Building

I've only tested building this in Ubuntu 22 in WSL2 on Windows 10. If you have issues building the UI code, either check the UI build instructions in the section [Running In Ubuntu WSL2](ui/README.md#running-in-ubuntu-wsl2) or comment out the `add_subdirectory` in the top level `CMakeLists.txt`.

```
mkdir build
cd build
cmake ../
cmake --build .
```