import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import logging
logging.getLogger().setLevel(logging.INFO)

from data import load_data
from parameters import *
from models_explicit import GenericModel

def main():
    data = load_data(dataset_path)

    model = GenericModel()
    model.make_model()
    graph = model.get_graph()

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        session.run(model.running_vars_initializer)
        for step in range(num_steps):
            # do I need to randomly sample?
            offset = (step * batch_size) % (data.train._labels.shape[0] - batch_size)
            minibatch_data = data.train._features[offset:(offset + batch_size), :]
            minibatch_labels_probas = data.train._labels_probas[offset:(offset + batch_size)]
            minibatch_labels = data.train._labels[offset:(offset + batch_size)]

            feed_dict = {model.tf_train_dataset : minibatch_data, model.tf_train_labels : np.array(minibatch_labels), model.tf_train_labels_probas: np.array(minibatch_labels_probas)}

            _, l, predictions = session.run([model.optimizer, model.loss, model.train_prediction()], feed_dict = feed_dict)
            # returns loss on training set
            pred_bin = get_binary_outputs(predictions)

            session.run(model.acc_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})
            session.run(model.recall_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred: pred_bin})
            session.run(model.precision_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})

            if step % display_step == 0:
                print 'Minibatch loss at step {0}: {1}'.format(step, l)
                accuracy = session.run(model.acc)
                recall = session.run(model.recall)
                precision = session.run(model.precision)
                print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)

            #w = session.run(model.weights)
            #b = session.run(model.biases)
            #print w1
            #print w1.shape
            #print w2
            #print w2.shape

        #print 'Test accuracy: {0}%'.format(accuracy(test_prediction.eval(), test_labels))
        # Pass the variables as a dict:
        #saver = tf.train.Saver({'v1': model.layer1_weights, 'v2': model.layer2_weights})
        for a in range(len(model.weights)):
            w = session.run(model.weights[a])
            b = session.run(model.biases[a])
            model.save_weights(w,'w'+str(a)+identifier)
            model.save_weights(b,'b'+str(a)+identifier)
            w_l = model.load_weights('w'+str(a)+identifier)
            print w_l.shape
            b_l = model.load_weights('b'+str(a)+identifier)
            print b_l.shape

        # perform prediction on some test set

        minibatch_data = data.test._features[0:batch_size,:]
        P = session.run(model.predict_op, feed_dict={model.tf_train_dataset: minibatch_data})
        print P
        loss = session.run(model.loss, feed_dict={model.model_scores: P, model.tf_train_labels_probas: data.test._labels_probas[0:batch_size]})
        print loss

        accuracy, recall, precision = get_metrics(predictions, true_labels, session, model)


def get_binary_outputs(predictions_proba):
    pred_binary = []
    for p in predictions_proba:
        if p[0]>=0.5:
            pred_binary.append(0)
        else:
            pred_binary.append(1)

    return np.array(pred_binary)


def get_metrics(predictions, true_labels, session, model):
    pred_bin = get_binary_outputs(predictions)
    session.run(model.acc_op, feed_dict = {model.tf_train_labels: true_labels, model.tf_train_labels_pred:  pred_bin})
    session.run(model.recall_op, feed_dict = {model.tf_train_labels: true_labels, model.tf_train_labels_pred:  pred_bin})
    session.run(model.precision_op, feed_dict = {model.tf_train_labels: true_labels, model.tf_train_labels_pred:  pred_bin})

    accuracy = session.run(model.acc)
    recall = session.run(model.recall)
    precision = session.run(model.precision)
    print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
    return accuracy, recall, precision

################
if __name__=='__main__':
    main()
