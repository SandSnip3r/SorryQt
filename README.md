# Qt Sorry

This repository provides a UI for the board game Sorry along with visualization of a Monte Carlo Tree Search bot which can be played against as an opponent.

Make sure to clone the submodule for the Sorry and MCTS implementations.

![img](appSorry_D5TMKN3cNc.png)

# Building in Qt Creator

1. Open project in Qt Creator by selecting the CMakeLists.txt
2. Select a kit (this includes selecting a Qt version and compiler)
3. Click the green arrow to build & run

# Building on command line

```
mkdir build
cd build
cmake ../
cmake --build .
```