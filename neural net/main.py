import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import logging
logging.getLogger().setLevel(logging.INFO)
import os

from data import load_data
from parameters import *
from models_explicit import GenericModel

def main():
    # Load dataset
    data = load_data(dataset_path, normalised=normalL)

    # Make directory if it doesn't exist
    if not os.path.exists(identifier):
        os.makedirs(identifier)

    # Instanciate models
    model = GenericModel()
    model.make_model()
    graph = model.get_graph()

    # Start training
    best_loss  = 1000
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        session.run(model.running_vars_initializer)
        for step in range(num_steps):

            # Setting up batch learning parameters
            offset = (step * batch_size) % (data.train._labels.shape[0] - batch_size)
            minibatch_data = data.train._features[offset:(offset + batch_size), :]
            minibatch_labels_probas = data.train._labels_probas[offset:(offset + batch_size)]
            minibatch_labels = data.train._labels[offset:(offset + batch_size)]

            # training dictionary
            feed_dict = {model.tf_train_dataset : minibatch_data, model.tf_train_labels : np.array(minibatch_labels), model.tf_train_labels_probas: np.array(minibatch_labels_probas)}

            # Start training
            _, l, predictions = session.run([model.optimizer, model.loss, model.train_prediction()], feed_dict = feed_dict)
            
            # Return loss on training set
            pred_bin = get_binary_outputs(predictions)

            # Fetch parameters to be able to print them
            session.run(model.acc_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})
            session.run(model.recall_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred: pred_bin})
            session.run(model.precision_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})

            if step % display_step == 0:
                print 'Minibatch loss at step {0}: {1}'.format(step, l)
                accuracy = session.run(model.acc)
                recall = session.run(model.recall)
                precision = session.run(model.precision)
                print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                if (l < best_loss) and step > 5000:
                    for indx in range(len(model.weights)):
                        w = session.run(model.weights[indx])
                        b = session.run(model.biases[indx])
                        model.save_weights(w,identifier+'best_w'+str(indx))
                        model.save_weights(b,identifier+'best_b'+str(indx))
                    best_loss = l

        # After training, save last iteration of the model
        for indx in range(len(model.weights)):
            w = session.run(model.weights[indx])
            b = session.run(model.biases[indx])
            model.save_weights(w,identifier+'w'+str(indx))
            model.save_weights(b,identifier+'b'+str(indx))

        # Perform prediction on some test set
        # and compute  average recall, precison and accuracy

        nfolds  = data.validation._labels.shape[0]/batch_size
        accuracy_accum = []
        recall_accum = []
        precision_accum = []
        print nfolds
        for fold in range(nfolds):
            minibatch_data = data.validation._features[fold*batch_size:(fold+1)*batch_size,:]
            minibatch_labels = data.validation._labels[fold*batch_size:(fold+1)*batch_size]
            #print np.sum(minibatch_labels)
            P = session.run(model.predict_op, feed_dict={model.tf_train_dataset: minibatch_data})
            [accuracy, recall, precision] = get_metrics(P, minibatch_labels, session, model)
            accuracy_accum.append(accuracy)
            recall_accum.append(recall)
            precision_accum.append(precision)

    print('Accuracy: %f, Recall: %f, Precision: %f' %(np.mean(accuracy_accum), np.mean(recall_accum), np.mean(precision_accum)))

def get_binary_outputs(predictions_proba):
    pred_binary = []
    """ Generate labels based on probability estimates"""
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
