import sys
sys.path.append('..')
from utils import *

import tensorflow as tf

class ChessNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        # print("1")

        # Renaming functions
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.layers.batch_normalization
        Dropout = tf.layers.dropout
        Dense = tf.layers.dense
        # print("2")
        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default():
            # print("3")
            self.input_boards = tf.placeholder(tf.float32, shape=[None, self.board_x, self.board_y])    # s: batch_size x board_x x board_y
            # print("4")
            self.dropout = tf.placeholder(tf.float32)
            # print("5")
            self.isTraining = tf.placeholder(tf.bool, name="is_training")
            # print("6")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 1])                    # batch_size  x board_x x board_y x 1
            # print("7")
            h_conv1 = Relu(BatchNormalization(self.conv2d(x_image, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            # print("8")
            h_conv2 = Relu(BatchNormalization(self.conv2d(h_conv1, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            # print("9")
            h_conv3 = Relu(BatchNormalization(self.conv2d(h_conv2, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels

            h_conv4 = Relu(BatchNormalization(self.conv2d(h_conv3, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels

            h_conv5 = Relu(BatchNormalization(self.conv2d(h_conv4, args.num_channels, 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels

            h_conv6 = Relu(BatchNormalization(self.conv2d(h_conv5, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
            # print("10")
            h_conv7 = Relu(BatchNormalization(self.conv2d(h_conv6, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
            # print("11")
            h_conv8 = Relu(BatchNormalization(self.conv2d(h_conv7, args.num_channels, 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
            # print("12")
            h_conv8_flat = tf.reshape(h_conv8, [-1, args.num_channels*(self.board_x-6)*(self.board_y-6)])
            # print("13")
            s_fc1 = Dropout(Relu(BatchNormalization(Dense(h_conv8_flat, 1024), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            # print("14")
            s_fc2 = Dropout(Relu(BatchNormalization(Dense(s_fc1, 512), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            # print("15")
            self.pi = Dense(s_fc2, self.action_size)                                                        # batch_size x self.action_size
            # print("16")
            self.prob = tf.nn.softmax(self.pi)
            # print("17")
            self.v = Tanh(Dense(s_fc2, 1))                                                               # batch_size x 1
            # print("18")
            self.calculate_loss()

    def conv2d(self, x, out_channels, padding):
      return tf.layers.conv2d(x, out_channels, kernel_size=[3,3], padding=padding)

    def calculate_loss(self):
        #print("1")
        self.target_pis = tf.placeholder(tf.float32, shape=[None, self.action_size]) #possible error
        self.target_vs = tf.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.losses.softmax_cross_entropy(self.target_pis, self.pi)#changed according to github
        self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,])) #new error
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.train.AdamOptimizer(self.args.lr).minimize(self.total_loss)
