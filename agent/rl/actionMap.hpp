#ifndef ACTION_MAP_HPP_
#define ACTION_MAP_HPP_

#include <sorry/engine/action.hpp>

#include <functional>
#include <vector>

class ActionMap {
public:
  static const ActionMap& getInstance() {
    static ActionMap instance;
    return instance;
  }
  
  sorry::engine::Action indexToAction(int index) const;
private:
  ActionMap();

  struct ActionRange {
    ActionRange(size_t start, size_t end, std::function<sorry::engine::Action(size_t)> actionGenerator)
      : start(start), end(end), actionGenerator(actionGenerator) {}
    size_t start;
    size_t end;
    std::function<sorry::engine::Action(size_t)> actionGenerator;
  };

  std::vector<ActionRange> actionRanges_;
};

#endif // ACTION_MAP_HPP_