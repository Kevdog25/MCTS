from Connect4MCTS import Connect4MCTS as AI
import numpy as np


def playGame(ai, playHuman = False, timePerMove = None, playsPerMove = None):
    board = ai.NewGame()
    print(board)
    print()
    ai.ResetRoot()
    while ai.Winner(board) is None:
        board = ai.FindMove(board, timePerMove, playsPerMove)
        print(board)
        print('{0}'.format(ai.Root.ChildWinRates()))
        print('Number of simulations: {0}'.format(ai.Root.Plays))
        print()
        ai.MoveRoot(board)
        if playHuman:
            col = int(input('Select a column: '))
            board = ai.ApplyAction(board,col)
            print()
            ai.MoveRoot(board)

    return


def addRootDist(root, plays):
    plays.append(root.Plays)
    if root.Children is not None:
        for c in root.Children:
            if c is not None:
                addRootDist(c, plays)
    return

if __name__=='__main__':
    np.set_printoptions(formatter={'float_kind': lambda x : "%.1f" % x})
    ai = AI(10, 1)
    playGame(ai, True, None, 100)

