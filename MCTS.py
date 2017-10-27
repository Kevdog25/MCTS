import numpy as np
import time
import threading

class Node:
    def __init__(self, state, legalActions, priors, **kwargs):
        self.State = state
        self.Wins = 0
        self.Plays = 0
        self.LegalActions = legalActions
        self.Children = None
        self.Parent = None
        # Use the legal actions mask to ignore priors that don't make sense.
        self.Priors = np.multiply(priors, legalActions) 
        
        return super().__init__(**kwargs)

    def Tally(self, val):
        self.Wins += val
        self.Plays += 1
        return

    def WinRate(self):
        return self.Wins/self.Plays if self.Plays > 0 else 0.

    def ChildWinRates(self):
        return np.array([n.WinRate() if n is not None else 0. for n in self.Children])

    def ChildPlays(self):
        return np.array([n.Plays if n is not None else 0. for n in self.Children])

class MCTS:
    '''This is a base class for Monte Carlo Tree Search algorithms. It outlines all the necessary operations for the core algorithm. 
        Most operations will need to be overriden to avoid a NotImplemenetedError.'''
    def __init__(self, timeLimit, explorationRate, threads = 1, **kwargs):
        self.TimeLimit = timeLimit
        self.ExplorationRate = explorationRate
        self.Root = None
        self.Threads = threads
        return super().__init__(**kwargs)

    def FindMove(self, state, moveTime = None):
        '''Given a game state, this will use a Monte Carlo Tree Search algorithm to pick the best next move.'''
        if moveTime is None:
            moveTime = self.TimeLimit
        endTime = time.time() + moveTime

        if self.Root is None:
            self.Root = Node(state, self.LegalActions(state), self.GetPriors(state))
        assert self.Root.State == state, 'MCTS has been primed for the correct input state.'

        if self.Threads == 1:
            self._runMCTS(self.Root, endTime)
        elif self.Threads > 1:
            self._runAsynch(state,endTime)

        return self.ApplyAction(state, self.SelectAction(self.Root, True))

    def _runAsynch(self, state, endTime):
        roots = []
        jobs = []
        for i in range(self.Threads):
            root = Node(state, self.LegalActions(state), self.GetPriors(state))
            roots.append(root)
            jobs.append(threading.Thread(None, target = self._runMCTS, args = (root, endTime)))
            jobs[-1].start()

        for j in jobs:
            j.join()

        self._mergeAll(self.Root, roots)
        return

    def _runMCTS(self, root, endTime):
        while time.time() < endTime:
            self.RunSimulation(root)
        return 

    def _mergeAll(self, target, trees):
        for t in trees:
            target.Plays += t.Plays
            target.Wins += t.Wins
        
        continuedTrees = [t for t in trees if t.Children is not None]
        if len(continuedTrees) == 0:
            return
        if target.Children is None:
            t = continuedTrees[0]
            target.Children = t.Children
            t.Children = None
            for c in target.Children:
                c.Parent = target
            del continuedTrees[0]

        for i in range(len(target.Children)):
            if target.Children[i] is None:
                continue
            self._mergeAll(target.Children[i], [t.Children[i] for t in continuedTrees])

        
        return

    def SelectAction(self, root, testing = False):
        '''Selects a child of the root using an upper confidence interval. If you are not exploring, setting the testing flag will 
            instead choose the one with the highest expected payouot - ignoring the exploration/regret factor.'''
        assert root.Children is not None, 'The node has children to select.'

        upperConfidence = root.ChildWinRates()
        if not testing:
            upperConfidence += self.ExplorationRate * root.Priors * np.sqrt(root.Plays) / (1.0 + root.ChildPlays())

        return np.argmax(upperConfidence + root.LegalActions)

    def AddChildren(self, node):
        '''Expands the node and adds children, actions and priors.'''
        l = len(node.LegalActions)
        node.Children = [None] * l
        for i in range(l):
            if node.LegalActions[i] == 1:
                s = self.ApplyAction(node.State, i)
                node.Children[i] = Node(s, self.LegalActions(s), self.GetPriors(s))
                node.Children[i].Parent = node
        return

    def MoveRoot(self, states):
        for s in states: 
            self._moveRoot(s)
        return

    def _moveRoot(self, state):
        if self.Root is None:
            return
        if self.Root.Children is None:
            self.Root = None
            return
        for child in self.Root.Children:
            if child == None:
                continue
            if child.State == state:
                self.Root = child
                break
        return

    def ResetRoot(self):
        if self.Root is None:
            return
        while self.Root.Parent is not None:
            self.Root = self.Root.Parent
        return

    def DropRoot(self):
        self.Root = None
        return

    def BackProp(self, leaf, stateValue):
        leaf.Tally(stateValue)
        if leaf.Parent is not None:
            self.BackProp(leaf.Parent, 1 - stateValue) # This could be better. If we checked the player instead of just flipping it.
        return
    
    def RunSimulation(self, root):
        raise NotImplementedError
    
    def SampleValue(self, state, player):
        raise NotImplementedError

    def GetPriors(self, state):
        raise NotImplementedError

    '''Game implementation functions.'''
    def ApplyAction(self, state, action):
        raise NotImplementedError

    def LegalActions(self, state):
        raise NotImplementedError

    def Winner(self, state, lastAction = None):
        raise NotImplementedError

if __name__=='__main__':
    mcts = MCTS(1, np.sqrt(2))
    print(mcts.TimeLimit)