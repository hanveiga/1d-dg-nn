from parameters2D import *
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

class GenericModel(object):
        def __init__(self):
            # start model
            self.graph = tf.Graph()
            self.weights = []
            self.biases = []
            # maybe refactor later

        def make_model(self):
            batch_size = 128

            self.input_layer = [num_input] + hidden_layers  #[num_features, hidden_nodes_1, hidden_nodes_2] #, hidden_nodes_3, hidden_nodes_4]
            self.output_layer = hidden_layers + [num_classes] #[hidden_nodes_1, hidden_nodes_2,num_labels] #, hidden_nodes_3, hidden_nodes_4, num_labels]

            with self.graph.as_default():
                self.tf_train_dataset = tf.placeholder(tf.float64, shape = (batch_size, num_input))
                self.tf_train_labels = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_train_labels_probas = tf.placeholder(tf.float64, shape = (batch_size,num_classes))

                self.tf_train_labels_pred = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_train_labels_probas_pred = tf.placeholder(tf.float64, shape = (batch_size,num_classes))


                self.tf_test_dataset = tf.placeholder(tf.float64, shape = (batch_size, num_input))
                self.tf_test_labels = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_test_labels_probas = tf.placeholder(tf.float64, shape = (batch_size,num_classes))
                self.tf_test_labels_pred = tf.placeholder(tf.float64, shape = (batch_size))
                self.tf_test_labels_probas_pred = tf.placeholder(tf.float64, shape = (batch_size,num_classes))

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
                self.loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.tf_train_labels_probas, dtype=tf.float64),self.model_scores, pos_weight=pos_weight))

                # Optimizer
                #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

                # Predictions
                #test_prediction = tf.nn.softmax(self.four_layer_network_(tf_test_dataset))
                #self.predict_op = tf.nn.sigmoid(self.four_layer_network_(self.tf_train_dataset))
                self.predict_op = tf.nn.softmax(self.four_layer_network_(self.tf_train_dataset))

                self.acc, self.acc_op = tf.metrics.accuracy(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
                self.recall, self.recall_op = tf.metrics.recall(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
                self.precision, self.precision_op = tf.metrics.precision(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")

                # Evaluate on test set

                #targets=tf.cast(self.tf_train_labels_probas, dtype=tf.float64)
                self.model_test_scores = self.four_layer_network_(self.tf_test_dataset)
                self.loss_test = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.tf_test_labels_probas, dtype=tf.float64),self.model_test_scores, pos_weight=pos_weight)))
                self.acc_test, self.acc_test_op = tf.metrics.accuracy(labels=self.tf_test_labels, predictions=self.tf_test_labels_pred, name="accuracy_test")
                self.recall_test, self.recall_test_op = tf.metrics.recall(labels=self.tf_test_labels, predictions=self.tf_test_labels_pred, name="accuracy_test")
                self.precision_test, self.precision_test_op = tf.metrics.precision(labels=self.tf_test_labels, predictions=self.tf_test_labels_pred, name="accuracy_test")


                # Isolate the variables stored behind the scenes by the metric operation
                running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
                running_vars2 = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_test")

                # Define initializer to initialize/reset running variables
                self.running_vars_initializer = tf.variables_initializer(var_list=running_vars) #,running_vars2))
                self.running_vars_initializer2 = tf.variables_initializer(var_list=running_vars2)

        def train_prediction(self):
            return tf.nn.softmax(self.model_scores)

        def test_prediction(self):
            return tf.nn.softmax(self.four_layer_network_(self.tf_test_dataset))

        def four_layer_network_(self, data):
            """
            chain up fn(...f1((f0(x) + b0)+b1) + bn
            """
            previous_layer = tf.matmul(data, self.weights[0])

            for hidden in range(len(self.weights)-1):
                hidden_layer = tf.nn.relu(previous_layer + self.biases[hidden])
                previous_layer = tf.matmul(hidden_layer, self.weights[hidden+1])

            output_layer = previous_layer + self.biases[len(self.biases)-1]

            return output_layer

        def save_weights(self, weight, filename):
            print weight.shape
            np.savetxt(filename+'.txt',weight,delimiter=';')

        def load_weights(self, filename):
            return np.loadtxt(filename+'.txt',dtype=np.float32,delimiter=';')

        def get_graph(self):
            return self.graph

        def save_all_weights(self, session):
            pass

class LoadedModel(GenericModel):

    def make_model(self, folder_path):
        batch_size = 128

        self.input_layer = [num_input] + hidden_layers  #[num_features, hidden_nodes_1, hidden_nodes_2] #, hidden_nodes_3, hidden_nodes_4]
        self.output_layer = hidden_layers + [num_classes] #[hidden_nodes_1, hidden_nodes_2,num_labels] #, hidden_nodes_3, hidden_nodes_4, num_labels]

        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float64, shape = (batch_size, num_input))
            self.tf_train_labels = tf.placeholder(tf.float64, shape = (batch_size))
            self.tf_train_labels_probas = tf.placeholder(tf.float64, shape = (batch_size,num_classes))

            self.tf_train_labels_pred = tf.placeholder(tf.float64, shape = (batch_size))
            self.tf_train_labels_probas_pred = tf.placeholder(tf.float64, shape = (batch_size,num_classes))

            # Initialize weights and biases for computation graph
            labels = [str(a) for a in range(len(self.output_layer))]
            for input_layer, output_layer, label in zip(self.input_layer,self.output_layer,labels):
                #self.weights.append(tf.Variable(tf.truncated_normal([input_layer, output_layer],dtype=tf.float64),name=label))
                #self.biases.append(tf.Variable(tf.zeros([output_layer],dtype=tf.float64)))
                a = self.load_weights(identifier+'best_w'+str(label))
                b = self.load_weights(identifier+'best_b'+str(label))
                self.weights.append(tf.Variable(tf.convert_to_tensor(a,dtype=tf.float64),name=label))
                self.biases.append(tf.Variable(tf.convert_to_tensor(b,dtype=tf.float64)))

            # Generate network
            self.model_scores = self.four_layer_network_(self.tf_train_dataset)

            # Loss function
            #self.loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.model_scores, labels=tf.cast(self.tf_train_labels, dtype=tf.int32, weights=[1.0,10.0])))

            # weighted class loss
            targets=tf.cast(self.tf_train_labels_probas, dtype=tf.float64)
            self.loss = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(tf.cast(self.tf_train_labels_probas, dtype=tf.float64),self.model_scores, pos_weight=pos_weight)))

            # Optimizer
            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            # Predictions
            #test_prediction = tf.nn.softmax(self.four_layer_network_(tf_test_dataset))
            #self.predict_op = tf.nn.sigmoid(self.four_layer_network_(self.tf_train_dataset))
            self.predict_op = tf.nn.softmax(self.four_layer_network_(self.tf_train_dataset))

            self.acc, self.acc_op = tf.metrics.accuracy(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
            self.recall, self.recall_op = tf.metrics.recall(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")
            self.precision, self.precision_op = tf.metrics.precision(labels=self.tf_train_labels, predictions=self.tf_train_labels_pred, name="accuracy")

            # Isolate the variables stored behind the scenes by the metric operation
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")

            # Define initializer to initialize/reset running variables
            self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)
#def save_weights(weight, filename):
#    np.savetxt(filename+'.txt',weight,delimiter=';')
#def load_weights(filename):
#    return np.loadtxt(filename+'.txt',dtype=np.float32,delimiter=';')

def accuracy(predictions, labels):
    preds_correct_boolean =  np.argmax(predictions, 1) == np.argmax(labels, 1)
    correct_predictions = np.sum(preds_correct_boolean)
    accuracy = 100.0 * correct_predictions / predictions.shape[0]
    return accuracy
