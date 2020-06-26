# Minigrod

Minigrod is a blood bowl playing ai developed for the 2020 Bot Bowl competition. It is based on the 2019 winner, Grod Bot, but with a number of improvements.


## Improvements

### Actions taken in order of risk
The original Grodbot took the 'best' available action during each turn. Minigrod progresses from less risky options to more risky ones. Although a generally good heuristic (and one often taught to new players), this does mean it sometimes wastes the move of a player who could recover the ball or takes an uneccessary block before scoring.

### Seperation of risk and outcome
The original grodbot attempted to combine how risky a move is and how good it would be if that move succeeded into a single number early in the assessment process. Minigrod attempts to keep these seperate until the last possible moment (when they must be combined in order to select a 'best' move).

### Score value and calculation changes
The original grodbot scaled the score of a move with the square of the path risk and not at all with the risk of the action, while Minigrod scales it with only the risk, but is aware of both path risk (dodging/gfi) and action risk (double skulls/getting sent off by the ref). In addition, a number of weights have been changed, added, or removed. The way to calculate how dangerous a square is for the ball carrier and where the cage corners need to be has been changed (although it's still not very good), block weights, mark weights, and a many other factors have been changed.

#### Standing up and doing nothing
Minigrod considers not moving or standing up without moving as potential actions.

#### Fouling
Minigrod fouls more than Grodbot.

#### Assisting
Minigrod moves players to assist squares to allow two or three dice blocks.

#### Receiving
Minigrod does not intentionally send players downfield to receive the ball.

#### Picking up the ball
Minigrod will try to secure the ball before picking it up.

#### Scoring
Minigrod will more aggressively try to score, though still not as aggressively as a human player would during turns 7 and 8.

### Reroll management
Rerolls are used more aggressively the more that are available.

### Debugging 
Minigrod prints a wonderful ascii representation of the field to help with finding and diagnosing bugs.

### Reserved Squares, Players, and Actions
Minigrod has a basic capability to 'reserve' certain squares, players, and actions, preventing it's intended blitz square from being blocked by another player or allowing a player to wait for handoff. This functionality could be expanded.

### Dynamic Offensive LOS
Minigrod customizes it's offensive line of scrimmage in an attempt to get the most two dice blocks. This could be improved upon.

## Potential Future Features

### Blitz To Start Move
Minigrod does not yet understand situations where it might be advantageous to blitz the ball carrier or another player free.

### Better screening, caging, and ball safety
Minigrod is unable to recognize what parts of the field can be secured with a screen or a cage, where to put the ball carrier, or whether a corner of the cage is needed.

### Move Importance
Minigrod cannot tell how 'important' a move is to its general goal. It needs to be able to tell the difference between a critical dodge with the ball carrier and meaningless 2 dice block on the other side of the field at the end of its turn.

### Better Scoring Solutions
Minigrod considers a very limited range of scoring options. Either running into the endzone or handing off/passing to someone else in range. It needs to consider additional moves to free the ball carrier or recievers.

### Better chain pushing
Chain pushing is something computers should be much better at than people, but Minigrod currently makes no calculations regarding them at all.

### Faster
Minigrod currently takes about 2 seconds to find the 'best' move for eleven players. While fast by human standards, this limits the ability to employ learning algorithms or generate large amounts of play data. If this number could be made a hundred times faster or so, machine learning opportunites might open up.

### Genetic Algorithm Weighting
Minigrod's actions are essentially controlled by a number of fairly arbitrary weights. How much a touchdown is worth, how good a two dice block is, etc. The optimal number for those weights is likely quite different than what they are currently set at and could use human or machine improvement.
