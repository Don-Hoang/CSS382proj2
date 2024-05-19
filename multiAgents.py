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

# CSS 382 Project 2
# Authors: Don Hoang, James Woo
# Date: 5/18/2024

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodLeft = newFood.count()
        closestFood = 9999999
        closestGhost = 9999999
        ghostScared = 0
        for food in newFood.asList():
            dist = manhattanDistance(newPos, food)
            if dist < closestFood:
                closestFood = dist
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if dist < closestGhost:
                closestGhost = dist
        if closestGhost is True:
            return successorGameState.getScore() - (1.0 / closestGhost)
        else:
            return successorGameState.getScore() + (1.0 / closestFood)

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        index = 0
        max = -9999999
        finalAction = None
        def Minimax(gameState, index, currDepth):
            # Check if we have reached the last ghost
            if index == gameState.getNumAgents():
                index = 0
                currDepth += 1
            # Check if the game is over or if we have reached the depth
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            # If it is pacman's turn
            if index == 0:
                max = -9999999
                action = None 
                moves = gameState.getLegalActions(index)
                for move in moves:
                    nextState = gameState.generateSuccessor(index, move)
                    score = Minimax(nextState, index + 1, currDepth)
                    if score > max:
                        max = score
                        action = move
                return max
            # If it is the ghost's turn
            else:
                min = 9999999
                action = None
                moves = gameState.getLegalActions(index)
                for move in moves:
                    state = gameState.generateSuccessor(index, move)
                    temp = Minimax(state, index + 1, currDepth)
                    if temp < min:
                        min = temp
                        action = move
                return min
        # Get the legal moves for pacman
        finalMove = gameState.getLegalActions(index)
        for move in finalMove:
            nextState = gameState.generateSuccessor(index, move)
            score = Minimax(nextState, index + 1, 0)
            if score > max:
                max = score
                finalAction = move
        return finalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        index = 0
        finalAction = None
        def Alphabeta(state, index, currDepth, alpha, beta):
            maxNumber = -9999
            minNumber = 9999
            action = None
            # Check if we have reached the last ghost
            if index == state.getNumAgents():
                index = 0
                currDepth += 1
            # Check if the game is over or if we have reached the depth
            if state.isWin() or state.isLose() or currDepth == self.depth:
                return self.evaluationFunction(state)
            # If it is pacman's turn
            if index == 0:
                action = None
                moves = state.getLegalActions(index)
                for move in moves:
                    tempState = state.generateSuccessor(index, move)
                    tempMax = Alphabeta(tempState, index + 1, currDepth, alpha, beta)
                    if tempMax > maxNumber:
                        maxNumber = tempMax
                        action = move
                    if maxNumber > beta:
                        return maxNumber
                    alpha = max(alpha, maxNumber)
                return maxNumber
            # If it is the ghost's turn
            else:
                action = None
                moves = state.getLegalActions(index)
                for move in moves:
                    tempState = state.generateSuccessor(index, move)
                    tempMin = Alphabeta(tempState, index + 1, currDepth, alpha, beta)
                    if tempMin < minNumber:
                        minNumber = tempMin
                        action = move
                    if minNumber < alpha:
                        return minNumber
                    beta = min(beta, minNumber)
                return minNumber
        # Get the legal moves for pacman
        alpha = -9999
        beta = 9999
        finalMove = gameState.getLegalActions(index)
        for move in finalMove:
            tempState = gameState.generateSuccessor(index, move)
            temp = Alphabeta(tempState, index + 1, 0, alpha, beta)
            if temp > alpha:
                alpha = temp
                finalAction = move
            if alpha > beta:
                break
        return finalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        index = 0
        finalAction = None
        max = -9999999

        def Expectimax(gameState, index, currDepth):
            # Check if we have reached the last ghost
            if index == gameState.getNumAgents():
                index = 0
                currDepth += 1
            # Check if the game is over or if we have reached the depth
            if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
                return self.evaluationFunction(gameState)
            # If it is pacman's turn
            if index == 0:
                max = -9999999
                action = None
                moves = gameState.getLegalActions(index)
                for move in moves:
                    nextState = gameState.generateSuccessor(index, move)
                    score = Expectimax(nextState, index + 1, currDepth)
                    if score > max:
                        max = score
                        action = move
                return max
            # If it is the ghost's turn
            else:
                total = 0
                action = None
                moves = gameState.getLegalActions(index)
                for move in moves:
                    state = gameState.generateSuccessor(index, move)
                    temp = Expectimax(state, index + 1, currDepth)
                    total += temp
                return total / len(moves)
        # Get the legal moves for pacman
        finalMove = gameState.getLegalActions(index)
        for move in finalMove:
            nextState = gameState.generateSuccessor(index, move)
            score = Expectimax(nextState, index + 1, 0)
            if score > max:
                max = score
                finalAction = move
        return finalAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    This function takes into account the following:
    - The closest food
    - The closest ghost
    - The number of pellets left
    - The score of the current game state and evaluates the game state based on these factors.
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    pellets = len(currentGameState.getCapsules())

    # Get the closest food
    foodClosest = 9999999
    ghostClosest = 9999999
    muliplierFood = 1
    muliplierGhost = 5
    muliplierPellets = 3

    # Get the closest food and ghosts
    for food in food.asList():
        dist = manhattanDistance(position, food)
        if dist < foodClosest:
            foodClosest = dist
    # Get the closest ghost
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(position, ghostPos)
        if dist < ghostClosest:
            ghostClosest = dist
    return currentGameState.getScore() + foodClosest * muliplierFood + \
        pellets * muliplierPellets - ghostClosest * muliplierGhost
   
# Abbreviation
better = betterEvaluationFunction
