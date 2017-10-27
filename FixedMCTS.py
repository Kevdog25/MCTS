from MCTS import MCTS
from MCTS import Node
import numpy as np

class FixedMCTS(MCTS):
    '''An implementation of Monte Carlo Tree Search that only aggregates statistics up to a fixed depth.'''
    def __init__(self, maxDepth, timeLimit, explorationRate, threads = 1, **kwargs):
        self.MaxDepth = maxDepth
        return super().__init__(timeLimit, explorationRate, threads, **kwargs)
    
    def RunSimulation(self, root):
        node = root
        depth = 0
        lastAction = None
        for i in range(self.MaxDepth):
            if node.Children is None:
                if self.Winner(node.State, lastAction) is not None:
                    break
                self.AddChildren(node)
            if np.sum(node.LegalActions) == 0:
                break
            previousPlayer = node.State.Player
            lastAction = self.SelectAction(node)
            node = node.Children[lastAction]

        assert i > 0, 'When requesting a move from the MCTS, there is at least one legal option.'

        self.BackProp(node, self.SampleValue(node.State, previousPlayer))
        return
    
    def SampleValue(self, state, player):
        '''Samples the value of the state for the specified player.'''
        rolloutState = state
        winner = self.Winner(rolloutState)
        while winner is None:
            actions = np.where(self.LegalActions(rolloutState) == 1)[0]
            action = np.random.choice(actions)
            rolloutState = self.ApplyAction(rolloutState, action)
            winner = self.Winner(rolloutState, action)
        return 0.5 if winner == 0 else int(winner == player)

    def GetPriors(self, state):
        return np.array([1 for v in self.LegalActions(state)])


