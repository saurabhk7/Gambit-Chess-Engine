from Coach import Coach
from chess.ChessGame import ChessGame as Game
from chess.tensorflow.NNet import NNetWrapper as nn
from utils import *
import sys
args = dotdict({
    'numIters': 1000, #original value: 1000
    'numEps': 100, #orginal value: 100
    'tempThreshold': 40, #15 #succesfully trained on tempThreshold: 3 ==> on Mar 28 2018
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200, #25x
    'counter':1,
    'arenaCompare': 40, #40
    'cpuct': 1,
    'checkpoint': './output',
    'load_model': False, #was False
    'load_folder_file': ('./output','temp.pth.tar'), #/dev/models/boardsize x numIters x numEps
    'numItersForTrainExamplesHistory': 20, #20

})

sys.setrecursionlimit(10**6)

if __name__=="__main__":
    g = Game(8) #returns the game object (constructor)
    nnet = nn(g) #NNet class returns NNetWrapper for the game object (g)
    print('----------------------********************-----------------------*********************-----------------')
    if args.load_model:
         nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    #
    print ('main.py==> ', 'args: ', args)
    c = Coach(g, nnet, args) #returns the Coach object with params(game_object, NeuralNet, argument values)
    if args.load_model:
        print("Load trainExamples from file")
        #c.loadTrainExamples()
    c.learn()
