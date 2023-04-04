# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


from math import sqrt
from functools import partial


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        # 10 points for every food you eat
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        newCapsule = successorGameState.getCapsules()
        # 200 points for every ghost you eat
        # but no point for capsule

        # For Ghost
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Position of ghost do not change regardless of your state
        # because you can't predict the future
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        # Count down from 40 moves
        ghostStartPos = [ghostState.start.getPosition() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        def dxdy(a, b):
            (ax, ay), (bx, by) = a, b
            return ax - bx, ay - by

        def manhaton(a, b):
            dx, dy = dxdy(a, b)
            return abs(dx) + abs(dy)

        def manhaton_cnt(a, b, th):
            return manhaton(a, b) > th

        def count(_, b, th):
            return b > th

        def euclidean(a, b):
            dx, dy = dxdy(a, b)
            return sqrt(dx**2 + dy**2)

        def dists(pacmanPos, posList, measure=manhaton):
            x, y = pacmanPos
            return [measure(pacmanPos, pos) for pos in posList]

        w1, w2, w3, w4, w5 = 1, 1, 1, 1, 5

        newWalls = successorGameState.getWalls()
        maxDist = newWalls.height + newWalls.width

        # food score
        foodPositions = newFood.asList()
        foodCount = newFood.count()
        foodDistance = dists(newPos, foodPositions, measure=manhaton)
        foodDistanceMin = min(foodDistance) / maxDist if foodCount > 0 else 0

        # capsule score
        capsuleCount = len(newCapsule)
        capsuleDistance = dists(newPos, newCapsule, measure=manhaton)
        capsuleDistanceMin = (min(capsuleDistance) if capsuleDistance else 0) / maxDist

        # ghost score
        ghostDistance = dists(newPos, ghostPositions, measure=partial(manhaton_cnt, th=1))
        ghostDistanceCnt = sum(ghostDistance)
        ghostScareTime = dists(newPos, newScaredTimes, measure=partial(count, th=1))
        ghostScareTimeCnt = sum(ghostScareTime)

        capsuleScore = -capsuleDistanceMin * w4 + -capsuleCount * w5
        foodScore = -foodCount * w1 + -foodDistanceMin * w2
        ghostScore = ghostDistanceCnt * w3 if not ghostScareTimeCnt else -ghostDistanceCnt * w3

        score = foodScore + ghostScore + capsuleScore + successorGameState.getScore()

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def value(self, gameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return "Stop", self.evaluationFunction(gameState)

        if self.isPacman(agentIndex):
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth)
            action_value[action] = value
        else:
            best = sorted(action_value.items(), key=lambda x: x[1], reverse=True)[0]
            action, value = best
            return action, value

    def min_value(self, gameState, agentIndex, depth):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth)
            action_value[action] = value
        else:
            best = sorted(action_value.items(), key=lambda x: x[1], reverse=False)[0]
            action, value = best
            return action, value

    def isPacman(self, agentIndex):
        return agentIndex == 0

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.agentNum = gameState.getNumAgents()
        action, _ = self.value(gameState, 0, self.depth)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def value(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return "Stop", self.evaluationFunction(gameState)

        if self.isPacman(agentIndex):
            return self.max_value(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.min_value(gameState, agentIndex, depth, alpha, beta)

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth, alpha, beta)
            if value > beta:
                return action, value
            else:
                action_value[action] = value
                alpha = max(alpha, value)
        else:
            best = sorted(action_value.items(), key=lambda x: x[1], reverse=True)[0]
            action, value = best
            return action, value

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth, alpha, beta)

            if value < alpha:
                return action, value
            else:
                action_value[action] = value
                beta = min(beta, value)
        else:
            best = sorted(action_value.items(), key=lambda x: x[1], reverse=False)[0]
            action, value = best
            return action, value

    def isPacman(self, agentIndex):
        return agentIndex == 0

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.agentNum = gameState.getNumAgents()
        alpha, beta = float("-inf"), float("inf")
        action, _ = self.value(gameState, 0, self.depth, alpha, beta)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def value(self, gameState, agentIndex, depth):
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return "Stop", self.evaluationFunction(gameState)

        if self.isPacman(agentIndex):
            return self.max_value(gameState, agentIndex, depth)
        else:
            return self.exp_value(gameState, agentIndex, depth)

    def max_value(self, gameState, agentIndex, depth):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth)
            action_value[action] = value
        else:
            best = sorted(action_value.items(), key=lambda x: x[1], reverse=True)[0]
            action, value = best
            return action, value

    def exp_value(self, gameState, agentIndex, depth):
        depth = depth - 1 if agentIndex == self.agentNum - 1 else depth
        legalMoves = gameState.getLegalActions(agentIndex)
        action_value = dict()
        for action in legalMoves:
            state = gameState.generateSuccessor(agentIndex, action)
            _, value = self.value(state, (agentIndex + 1) % self.agentNum, depth)
            action_value[action] = value
        else:
            exp = sum(action_value.values()) / len(action_value)
            # action, value = best
            return "Stop", exp

    def isPacman(self, agentIndex):
        return agentIndex == 0

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.agentNum = gameState.getNumAgents()
        action, _ = self.value(gameState, 0, self.depth)
        return action
