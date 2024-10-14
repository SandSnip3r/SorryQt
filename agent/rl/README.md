# Observation Space

This describes ideas for choosing an "observation space" for the Sorry game.

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

Since every piece will be somewhere, I think the piece oriented approach is less wasteful. A position oriented approach would have a lot of positions with no piece on them.

# Agent Performance

## Single Player - Random Agent

 - Green  - Average actions per game: 59.772336
 - Blue   - Average actions per game: 59.742279
 - Red    - Average actions per game: 59.745334
 - Yellow - Average actions per game: 59.757388