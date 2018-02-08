from parameters import *
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class GenericModel(object):
        def __init__(self):
            # start model
            self.graph = tf.Graph()
            self.weights = []
            self.biases = []
            # maybe refactor later

        def make_model(self):
            # a bit gross
            hidden_nodes_1 = n_hidden_1
            hidden_nodes_2 = n_hidden_2
            hidden_nodes_3 = n_hidden_3
            hidden_nodes_4 = n_hidden_4
            num_labels = num_classes
            batch_size = 128
            num_features = num_input
            learning_rate = 0.00001 # potentially adaptive learning rate

            self.input_layer = [num_features, hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, hidden_nodes_4]
            self.output_layer = [hidden_nodes_1, hidden_nodes_2, hidden_nodes_3, hidden_nodes_4, num_labels]

            with self.graph.as_default():
                self.tf_train_dataset = tf.placeholder(tf.float64, shape = (batch_size, num_features))
                self.tf_train_labels = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_train_labels_probas = tf.placeholder(tf.float64, shape = (batch_size,num_classes))

                self.tf_train_labels_pred = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_train_labels_probas_pred = tf.placeholder(tf.float64, shape = (batch_size,num_classes))


                # Initialize weights and biases for computation graph
                labels = [str(a) for a in range(len(self.output_layer))]
                for input_layer, output_layer, label in zip(self.input_layer,self.output_layer,labels):
                    self.weights.append(tf.Variable(tf.truncated_normal([input_layer, output_layer],dtype=tf.float64),name=label))
                    self.biases.append(tf.Variable(tf.zeros([output_layer],dtype=tf.float64)))

                # Generate network
                self.model_scores = self.four_layer_network_(self.tf_train_dataset)

                # Loss function
                #self.loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_scores, labels=tf.cast(self.tf_train_labels, dtype=tf.int32, weights=[1.0,10.0])))

                # weighted class loss
                targets=tf.cast(self.tf_train_labels_probas, dtype=tf.float64)
                #yout = tf.nn.softmax(self.model_scores)
                #self.loss = -tf.reduce_mean(targets*tf.log(yout)*tf.constant([1.,200.],dtype=tf.float64))
                #self.loss =  tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.model_scores,self.model_scores, pos_weight=10.0))
                #,targets=tf.cast(self.tf_train_labels, dtype=tf.int32)))
                #self.loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_scores, labels=tf.cast(self.tf_train_labels, dtype=tf.int32)))
                self.loss = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.tf_train_labels_probas, dtype=tf.float64),self.model_scores, pos_weight=10.0)))

                # Optimizer
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

                # Predictions
                #test_prediction = tf.nn.softmax(self.four_layer_network_(tf_test_dataset))
                self.predict_op = tf.nn.softmax(self.four_layer_network_(self.tf_train_dataset))

                self.acc, self.acc_op = tf.metrics.accuracy(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
                self.recall, self.recall_op = tf.metrics.recall(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
                self.precision, self.precision_op = tf.metrics.precision(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
                # Isolate the variables stored behind the scenes by the metric operation
                running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")

                # Define initializer to initialize/reset running variables
                self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)


        def train_prediction(self):
            # is this necessary?
            return tf.nn.softmax(self.model_scores)

        def four_layer_network_(self, data):
            """
            chain up fn(...f1((f0(x) + b0)+b1) + bn
            """

            input_layer = tf.matmul(data, self.weights[0])
            hidden_1 = tf.nn.relu(input_layer + self.biases[0])
            layer_2 = tf.matmul(hidden_1, self.weights[1])
            hidden_2 = tf.nn.relu(layer_2 + self.biases[1])
            layer_3 = tf.matmul(hidden_2, self.weights[2])
            hidden_3 = tf.nn.relu(layer_3 + self.biases[2])
            layer_4 = tf.matmul(hidden_3, self.weights[3])
            hidden_4 = tf.nn.relu(layer_4 + self.biases[3])

            output_layer = tf.matmul(hidden_4, self.weights[4]) + self.biases[4]

            return output_layer

        def save_weights(self, weight, filename):
            np.savetxt(filename+'.txt',weight,delimiter=';')

        def load_weights(self, filename):
            return np.loadtxt(filename+'.txt',dtype=np.float32,delimiter=';')

        def get_graph(self):
            return self.graph

        def save_all_weights(self, session):
            pass


def save_weights(weight, filename):
    np.savetxt(filename+'.txt',weight,delimiter=';')

def load_weights(filename):
    return np.loadtxt(filename+'.txt',dtype=np.float32,delimiter=';')

def accuracy(predictions, labels):
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy
