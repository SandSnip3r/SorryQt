#ifndef ACTION_MAP_HPP_
#define ACTION_MAP_HPP_

#include <sorry/engine/action.hpp>
#include <sorry/engine/playerColor.hpp>

#include <functional>
#include <optional>
#include <vector>

class ActionMap {
public:
  static const ActionMap& getInstance() {
    static ActionMap instance;
    return instance;
  }
  
  sorry::engine::Action indexToActionForPlayer(size_t index, sorry::engine::PlayerColor playerColor) const;
  size_t actionToIndex(sorry::engine::Action action) const;
  int totalActionCount() const;
private:
  ActionMap();

  struct ActionRange {
    using ActionGenerator = std::function<sorry::engine::Action(size_t, sorry::engine::PlayerColor)>;
    using IndexGenerator = std::function<std::optional<size_t>(const sorry::engine::Action&)>;
    ActionRange(size_t start, size_t end, ActionGenerator actionGenerator, IndexGenerator indexGenerator)
      : start(start), end(end), actionGenerator(actionGenerator), indexGenerator(indexGenerator) {}
    size_t start;
    size_t end;
    ActionGenerator actionGenerator;
    IndexGenerator indexGenerator;
  };

  std::vector<ActionRange> actionRanges_;
};

#endif // ACTION_MAP_HPP_