import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import numpy as np
from data2D import load_data, load_triangle_data
from parameters2D import *
from models_explicit2D import GenericModel, LoadedModel
from main2D import get_binary_outputs, get_metrics

#num_steps = 15000

def load_nn(folder_path):
    model = LoadedModel()
    model.make_model(identifier)
    return model

def main(folder_path):

    model = load_nn(folder_path)
    graph = model.get_graph()

    #data = load_data(dataset_path, normalised=normalL)
    data = load_triangle_data(target_data,dataset_path,normalised=normalL)


    #model = GenericModel()
    #model.make_model()
    #graph = model.get_graph()
    # retrain model
    trainingFlag = True
    if (trainingFlag == True):
        best_loss  = 1000
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            session.run(model.running_vars_initializer)
            for step in range(retrain_steps):

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

                if step % 1000 == 0:
                    print 'Minibatch loss at step {0}: {1}'.format(step, l)
                    accuracy = session.run(model.acc)
                    recall = session.run(model.recall)
                    precision = session.run(model.precision)
                    print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                    if (l < best_loss) and step > 4000:
                        for indx in range(len(model.weights)):
                            w = session.run(model.weights[indx])
                            b = session.run(model.biases[indx])
                            model.save_weights(w,identifier+'l1_retrained2_best_w'+str(indx))
                            model.save_weights(b,identifier+'l1_retrained2_best_b'+str(indx))
                        best_loss = l

                nfolds  = data.validation._labels.shape[0]/batch_size
                accuracy_accum = []
                recall_accum = []
                precision_accum = []

            for fold in range(nfolds):
                    minibatch_data = data.validation._features[fold*batch_size:(fold+1)*batch_size,:]
                    minibatch_labels = data.validation._labels[fold*batch_size:(fold+1)*batch_size]
                    #print np.sum(minibatch_labels)
                    P = session.run(model.predict_op, feed_dict={model.tf_train_dataset: minibatch_data})
                    [accuracy, recall, precision] = get_metrics(P, minibatch_labels, session, model)
                    print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                    accuracy_accum.append(accuracy)
                    recall_accum.append(recall)
                    precision_accum.append(precision)

    # validating loaded model
    if trainingFlag==False:
        best_loss  = 1000
        with tf.Session(graph=graph) as session:
                tf.global_variables_initializer().run()
                session.run(model.running_vars_initializer)
                nfolds  = data.validation._labels.shape[0]/batch_size
                accuracy_accum = []
                recall_accum = []
                precision_accum = []

                for fold in range(nfolds):
                    minibatch_data = data.validation._features[fold*batch_size:(fold+1)*batch_size,:]
                    minibatch_labels = data.validation._labels[fold*batch_size:(fold+1)*batch_size]
                    #print np.sum(minibatch_labels)
                    P = session.run(model.predict_op, feed_dict={model.tf_train_dataset: minibatch_data})
                    [accuracy, recall, precision] = get_metrics(P, minibatch_labels, session, model)
                    print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                    accuracy_accum.append(accuracy)
                    recall_accum.append(recall)
                    precision_accum.append(precision)

    #new_data = load_data(data_folder)
    #retrained_model = retrain_model(model,new_data)
    #validate_mode(retrained_model, data)

if __name__=='__main__':
    folder_path = identifier
    main(folder_path)
