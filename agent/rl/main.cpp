#include <sorry/common/common.hpp>
#include <sorry/engine/sorry.hpp>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// #include <pybind11/embed.h> // Everything needed for embedding

// namespace py = pybind11;

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <random>

/* General TODOs:
** - Sorry::reset() method
**/

using namespace std;

// How long does it take an agent acting randomly to finish a game of Sorry?
double simulateRandomGames() {
  constexpr const int kNumGames = 1'000'000;
  int totalActionCount = 0;
  mt19937 randomEngine = sorry::common::createRandomEngine();
  for (int i=0; i<kNumGames; ++i) {
    int thisGameActionCount = 0;
    sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kYellow});
    sorry.drawRandomStartingCards(randomEngine);
    while (!sorry.gameDone()) {
      const std::vector<sorry::engine::Action> actions = sorry.getActions();
      uniform_int_distribution<size_t> dist(0, actions.size() - 1);
      const sorry::engine::Action action = actions[dist(randomEngine)];
      sorry.doAction(action, randomEngine);
      ++thisGameActionCount;
    }
    totalActionCount += thisGameActionCount;
    if (i%1000 == 0) {
      cout << "Game " << i << ". Average actions per game: " << static_cast<double>(totalActionCount) / (i+1) << endl;
    }
  }
  return static_cast<double>(totalActionCount) / kNumGames;
}

namespace py = pybind11;

int main() {
  constexpr int kSeed = 0x5EED;
  mt19937 randomEngine{kSeed};

  // Initialize the Python interpreter
  // py::interp
  py::scoped_interpreter guard{};

  // Get the sys module
  py::module sys = py::module::import("sys");

  // Append the directory containing my_jax.py to sys.path, SOURCE_DIR is set from CMake.
  sys.attr("path").cast<py::list>().append(std::string(SOURCE_DIR));

  // Load the Python module
  py::module jax_module = py::module::import("my_jax");

  // Instantiate the MyModel class from Python
  py::object TestClass = jax_module.attr("TestClass");

  // Create an instance of MyModel
  py::object model_instance = TestClass();

  // Call function of class
  constexpr int kNumIterations = 10000;
  std::array<float, kNumIterations> results;
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  for (int i=0; i<kNumIterations; ++i) {
    const py::object result = model_instance.attr("func")(i);
    const auto &numpy_array = result.cast<py::array_t<float>>();
    const auto &data = numpy_array.unchecked<1>();
    results[i] = data(0);
  }
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  std::cout << "Results: [";
  for (int i=0; i<kNumIterations; ++i) {
    std::cout << results[i] << ", ";
  }
  std::cout << "]" << std::endl;
  std::cout << "Calculation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
  
  // Instantiate a Sorry game
  sorry::engine::Sorry sorry({sorry::engine::PlayerColor::kYellow});
  sorry.drawRandomStartingCards(randomEngine);
  // Construct an initial observation state
  // Pass the observation state to the neural network and get its action preference
  // Take the action
  return 0;
}