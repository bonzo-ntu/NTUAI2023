# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack, Queue, PriorityQueueWithFunction
from typing import Union, Optional, List, Dict, Set, Tuple


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class Node:
    problem = None
    explored = None
    fringe = None

    def __init__(self, state, parent_coor):
        """
        state: [coordinate, action, cost]
        """
        self.state = state
        self.coor, self.action, self.cost = self.state
        self.parent_coor = parent_coor

    @classmethod
    def set(cls, attr, value):
        if getattr(cls, attr) is None:
            setattr(cls, attr, value)

    @classmethod
    def setUp(
        cls,
        problem: Optional[SearchProblem],
        explored: Optional[Dict],
        fringe: Optional[Union[Stack, Queue, PriorityQueueWithFunction]],
    ):
        cls.set("problem", problem)
        cls.set("explored", explored)
        cls.set("fringe", fringe)

    @classmethod
    def tearDown(cls):
        del cls.problem
        del cls.explored
        del cls.fringe

        cls.problem = None
        cls.explored = None
        cls.fringe = None

    def isGoal(self):
        return self.problem.isGoalState(self.coor)

    def expand(self):
        successors = self.problem.getSuccessors(self.coor)
        return [Node(s, self.coor) for s in successors if s[0] not in self.explored]

    def push_to_explored(self):
        self.explored[self.coor] = (self.parent_coor, self.action)

    def fringe_push(self):
        self.fringe.push(self)

    def fringe_pop(self):
        return self.fringe.pop()

    @property
    def fringe_item(self):
        items = self.fringe.list
        return items

    def coor_in_fringe(self):
        return self.coor in [node.coor for node in self.fringe_item]

    def __str__(self):
        s = f"({self.coor} <=[{self.action}]= {self.parent_coor})| {[(_.coor, _.parent_coor) for _ in self.fringe_item]}"
        return s

    def back_track(self):
        coor = self.coor
        actions = []
        while self.explored[coor] != (None, None):
            parent_coor, action = self.explored[coor]
            actions.insert(0, action)
            coor = parent_coor
        return actions


class pqNode(Node):
    def __init__(self, state, parent_coor, parent_cost):
        """
        state: [coordinate, action, cost]
        """
        super().__init__(state, parent_coor)
        self.parent_cost = parent_cost

    def expand(self):
        successors = self.problem.getSuccessors(self.coor)
        return [pqNode(s, self.coor, self.cost_sofar) for s in successors if s[0] not in self.explored]

    @property
    def fringe_item(self):
        items = self.fringe.heap
        return items

    @property
    def cost_sofar(self):
        return self.cost + self.parent_cost

    def coor_in_fringe(self):
        # 當 cost 比較小的有
        return self.coor in [
            node.coor for priority, count, node in self.fringe_item if node.cost_sofar < self.cost_sofar
        ]

    def __str__(self):
        s = f"({self.coor} <=[{self.action}]= {self.parent_coor})| {[(_.coor, _.parent_coor) for priority, count, _ in self.fringe_item]}"
        return s

    def __repr__(self):
        s = f"<{self.state}, ({self.parent_coor}, {self.parent_cost})>"
        return s

    def back_track(self):
        coor = self.coor
        actions = []
        while self.explored[coor] != (None, None):
            parent_coor, action = self.explored[coor]
            actions.insert(0, action)
            coor = parent_coor
        return actions


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    coor = problem.getStartState()
    action = None
    cost = None
    state = (coor, action, cost)
    parent_coor = None
    current = Node(state, parent_coor)
    current.setUp(problem=problem, explored=dict(), fringe=util.Stack())
    while not current.isGoal():
        successors = current.expand()
        current.push_to_explored()
        [s.fringe_push() for s in successors]
        current = current.fringe_pop()
    else:
        # if current is goal, add current to explored
        current.push_to_explored()
        actions = current.back_track()

    current.tearDown()
    return actions


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    coor = problem.getStartState()
    action = None
    cost = None
    state = (coor, action, cost)
    parent_coor = None
    current = Node(state, parent_coor)
    current.setUp(problem=problem, explored=dict(), fringe=util.Queue())
    while not current.isGoal():
        successors = current.expand()
        current.push_to_explored()
        [s.fringe.push(s) for s in successors if not s.coor_in_fringe()]
        current = current.fringe_pop()
    else:
        # if current is goal, add current to explored
        current.push_to_explored()
        actions = current.back_track()

    current.tearDown()
    return actions


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    def realCost(node):
        return node.cost_sofar

    coor = problem.getStartState()
    action = None
    cost = 0
    state = (coor, action, cost)
    parent_coor = None
    parent_cost = 0
    current = pqNode(state, parent_coor, parent_cost)
    current.setUp(problem=problem, explored=dict(), fringe=util.PriorityQueueWithFunction(realCost))
    while not current.isGoal():
        successors = current.expand()

        current.push_to_explored()
        [s.fringe.push(s) for s in successors if not s.coor_in_fringe()]
        current = current.fringe_pop()
    else:
        # if current is goal, add current to explored
        current.push_to_explored()
        actions = current.back_track()

    current.tearDown()
    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    def realCost(node):
        return node.cost_sofar + heuristic(node.coor, node.problem)

    coor = problem.getStartState()
    action = None
    cost = 0
    state = (coor, action, cost)
    parent_coor = None
    parent_cost = 0
    current = pqNode(state, parent_coor, parent_cost)
    current.setUp(problem=problem, explored=dict(), fringe=util.PriorityQueueWithFunction(realCost))
    while not current.isGoal():
        successors = current.expand()

        current.push_to_explored()
        [s.fringe.push(s) for s in successors if not s.coor_in_fringe()]
        current = current.fringe_pop()
    else:
        # if current is goal, add current to explored
        current.push_to_explored()
        actions = current.back_track()

    current.tearDown()
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
