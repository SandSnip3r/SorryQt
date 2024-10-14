# Sorry

![board](images/board.png)

## Modified Rules

This section speaks about the rules of the "adult" version of the Sorry boardgame. First read [the original rules](https://www.hasbro.com/common/instruct/sorry.pdf) before reading this, as I will not repeat everything.

Sorry has a deck of 45 cards with 11 different card types: 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, and "Sorry". At all times, every player hold 5 cards. On a player's turn, a single card from their five must be played. The card the player chooses to play has the available actions written on the card; they're usually applied to one or two pieces on the board. Only if no card can be played, then the player may choose to discard any card. The player draws to replace the played card and then their turn is over.

There are a few discrepencies between different versions of Sorry. Different versions print different possible actions on their cards or have different rule descriptions. These rule differences are configurable in the `SorryRules` struct in `sorry.hpp`.

The main discrepencies are:
  1. The Sorry card may or may not alternatively allow the player to move forward 4 spaces.
  2. The 2 card may or may not grant the player an additional turn after playing, discarding, and replacing the 2.
  3. Sometimes the player starts with all 4 pieces in their "Start", other times, only 3 pieces in "Start" and the 4th piece is on the position immediately outside of their "Start".

### Scoring

The "variation for adules" version lists some scoring rules. We do not use these. We simply say that the first player to move all 4 pieces into the "Home" position has won. The second player to do so gets 2nd place and so on.

The rules also state that the game is for 2 to 4 players, but we allow for a single player version. The objective of the solo game is simply to move all 4 pieces into the "Home" position using as few cards as possible.