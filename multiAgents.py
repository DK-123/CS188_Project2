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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        ''' 
        Notes
        
        from the assignment spec:
        Note: Remember that newFood has the function asList()

        Note: As features, try the reciprocal of important values (such as distance to food) rather than 
        just the values themselves.

        - use floats
        - have 2 main components (food and ghost)
        - food component - reward being close to the nearest food pellet
            Using reciprocal of distance so closer food = much higher score 
            (aka being 1 away gives +1.0, being 10 away gives +0.1)
            Calculate distance to each food pellet and find minimum distance

        - ghost component - penalize being close to active ghosts and reward being close to scared ghosts (we can eat them)
        '''

        score =  successorGameState.getScore()

        # print(score)
        # print(newPos)
        # print(newFood)
        # print(newGhostStates)
        # print(newScaredTimes)

        # first handle food condition
        foodList = newFood.asList()
        if foodList:
            distances = []

            for food in foodList:
                dist = manhattanDistance(newPos, food)
                distances.append(dist)

            minFoodDist = min(distances)
            score += 1.0 / minFoodDist

        # next handle the ghosts
        for i in range(len(newGhostStates)):
            ghostState = newGhostStates[i]
            scaredTime = newScaredTimes[i]
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0 and dist > 0:
                score += 20.0 / dist
            else:
                score -= 600.0

        # I added this condition b/c otherwise sometimes Pacman gets stuck oscillating
        if action == Directions.STOP:
            score -= 10.0

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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
        actions = gameState.getLegalActions(0)
        next = []
        for action in actions:
            if action == actions[0]:
                next = [action]
            successor = gameState.generateSuccessor(curr_agent, next)
            curr_agent += 1
            if successor.isWin():
                next += [actions]
            if self.depth == tot_agents:
                return next
        while (self.depth )
        


        util.raiseNotDefined()
    def minimax_helper(index, gameState, action):
        agent_index = self.depth (gamestate())
        successor
        return 


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        '''
        state = the current game state (aka positions of Pacman, ghosts, food, walls)
        getLegalActions(0) gets all legal moves for agent 0 (Pacman)
        agentIndex=0 means Pacman, ghosts are >= 1
        '''
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.alphaBetaPruning(successor, self.depth, 1, alpha, beta)
            
            if score > bestScore:
                bestScore = score
                bestAction = action
            
            alpha = max(alpha, bestScore)
        
        return bestAction
    
    # recurisve func to check if we should stop searching aka if game is won/lost or depth limit reached
    def alphaBetaPruning(self, state, depth, agentIndex, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)
        
        if agentIndex == 0:
            return self.maxValue(state, depth, alpha, beta)
        else:
            return self.minValue(state, depth, agentIndex, alpha, beta)

    # helper function to handle max logic; I followed the pseudo-code algorithm from spec
    def maxValue(self, state, depth, alpha, beta):
        v = float('-inf')
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            v = max(v, self.alphaBetaPruning(successor, depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    # helper function to handle min logic using pseudo-code from spec 
    def minValue(self, state, depth, agentIndex, alpha, beta):
        v = float('inf')
        numAgents = state.getNumAgents()
        nextAgent = agentIndex + 1
        
        # this condition is to check if all agents have moved (aka nextAgent exceeds total agents)
        # if yes then we wrap to Pacman and decrease depth (1 round done)
        # otherwise we continue to next ghost at same depth
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth = depth - 1 
        else:
            nextDepth = depth
        
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            v = min(v, self.alphaBetaPruning(successor, nextDepth, nextAgent, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v
                    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        ''' 
        from spec:
        ExpectimaxAgent will no longer take the min over all ghost actions, but the expectation 
        according to your agent’s model of how the ghosts act
        '''
        bestAction = None
        bestScore = float('-inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.expectimax(successor, self.depth, 1)
            
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction
    
    # recursive func that returns a number representing the expected value of this state
    def expectimax(self, state, depth, agentIndex):        
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)   
        if agentIndex == 0:
            return self.maxValue(state, depth)
        else:
            return self.expectedValue(state, depth, agentIndex)
    
    # helper func to handle Pacman's turn (b/c Pacman wants to maximize his score by picking  best action)
    def maxValue(self, state, depth):
        v = float('-inf')
        
        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            v = max(v, self.expectimax(successor, depth, 1))
        
        return v
    
    # helper func to handle ghost turns (we assume ghosts move randomly) and returns expected score for Pacman
    def expectedValue(self, state, depth, agentIndex):
        numAgents = state.getNumAgents()
        nextAgent = agentIndex + 1
        legalActions = state.getLegalActions(agentIndex)
        total = 0
        
        if nextAgent == numAgents:
            nextAgent = 0
            nextDepth = depth - 1
        else:
            nextDepth = depth
        
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action)
            total += self.expectimax(successor, nextDepth, nextAgent)
        
        expectedScore = total / len(legalActions)
        
        return expectedScore

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
