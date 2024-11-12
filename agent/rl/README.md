# Observation Space

This section has ideas for choosing an "observation space" for the Sorry game.

## The Game

First, I describe what information is avaialble to every player in the game.

### Cards

Each player always holds exactly 5 cards. A player may see their cards, but not the cards of any other player. There are some number of face-down (unknown) cards. There are some number of discarded face-up (known) cards. There are 11 total card types; 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, and Sorry.

### Board Positions

All pieces/positions are visible to all players. The board has 60 "public" positions in a ring shape which all players move clockwise on. Each of these public positions can only have a single piece on it.
Each player also has a few positions which are private to them:
1. The "Start" position which can have anywhere from 0 to 4 of the player's pieces on it.
2. 5 positions which are the player's "safety" zone. Each of these positions may only have a single piece on it.
3. The "Home" position which can have anywhere from 0 to 4 of the player's pieces on it.

## Representation

Below are ideas how we could represent this information best for a neural network.

### Cards

The player's hand can be represented as 5 one-hot vectors of 11 items (one for each card type).

The discard pile can be represented as 40 one-hot vectors of 12 items (one for each card type and an additional item for "None"). I chose 40 because that is the largest the discard pile can be. Reminder that there are 45 cards and there must be at least one player.

_Should the card observation space explicitly say anything about the number of face-down cards?_

### Board Positions

I see two ways that we can think about the pieces on the board:
1. Position oriented - Every position has a description for which piece is on it.
2. Piece oriented - Every piece has a description for which position it is on.

Since every piece will be somewhere, I think the piece oriented approach is less wasteful. A position oriented approach would have a lot of positions with no piece on them. This would result in 4 one-hot vectors of 67 positions.

# Action Space

This section has ideeas for choosing an "action space" for the Sorry game.

Naively, you might think it is as simple as choosing a card from your hand and applying it to one of your pieces. However, there are a couple actions which require more information.

- We at least additionally need a discard action for each of your cards.
- The 7 card allows you to split your move across two pieces, so we need a way to specify this additional information.
- The 10 card has two possible actions, move forward 10 or move backward 1.
- The 11 card has two possible actions, move forward 11 or trace places with an opponent.

## Listed - All Possible Actions

Discard 1
Discard 2
...
Discard Sorry
(11 actions)

Play 1, move piece from 0 to 2
Play 1, move piece from 1 to 2
Play 1, move piece from 2 to 3
...
Play 1, move piece from 65 to 66
(66 actions)

Play 2, move piece from 0 to 2
Play 2, move piece from 1 to 3
Play 2, move piece from 2 to 4
...
Play 2, move piece from 64 to 66
(65 actions)

Play 3, move piece from 1 to 4
Play 3, move piece from 2 to 5
...
Play 3, move piece from 63 to 66
(63 actions)

Play 4, move piece from 1 to 57
Play 4, move piece from 2 to 58
Play 4, move piece from 3 to 59
Play 4, move piece from 4 to 60
Play 4, move piece from 5 to 1
...
Play 4, move piece from 65 to 61
(65 actions)

Play 5
(61 actions)

Play 7, as single move
(59 actions)
Play 7, as double move
(~11285 actions)

Play 8
(58 actions)

Play 10, forward 10
(56 actions)
Play 10, backward 1
(65 actions)

Play 11, forward 11
(55 actions)
Play 11, swap with opponent
(~58 actions)

Play 12
(54 actions)

# Agent Performance

_TODO_

# Installation

## Bug

pybind11-dev 2.9.1-2 is too old, try to get at least 2.11.1-2

I installed from source, which is 2.13 or so.
