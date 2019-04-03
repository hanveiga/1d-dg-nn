import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import numpy as np
from data2D import load_data, load_rd_quads
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
    alpha = 0.25
    data = load_rd_quads(target_quad_data,dataset_path, alpha = alpha ,normalised=normalL)


    #model = GenericModel()
    #model.make_model()
    #graph = model.get_graph()
    # retrain model
    trainingFlag = True
    if (trainingFlag == True):
        best_loss  = 1000
        best_test_loss  = 1000
        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            session.run(model.running_vars_initializer)
            session.run(model.running_vars_initializer2)
            nbatches = int(data.train._labels.shape[0]/batch_size)
            print nbatches
            nepocs = 5
            for epoch in range(nepocs):
                for step in range(nbatches):

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

                    if step % 50 == 0:
                        offset_test = (step * batch_size) % (data.test._labels.shape[0] - batch_size)
                        minibatch_data_t = data.test._features[offset_test:(offset_test + batch_size), :]
                        minibatch_labels_t_probas = data.test._labels_probas[offset_test:(offset_test + batch_size)]
                        minibatch_labels_t = data.test._labels[offset_test:(offset_test + batch_size)]

                        feed_dict_t = {model.tf_test_dataset : minibatch_data_t, model.tf_test_labels : np.array(minibatch_labels_t),\
                                       model.tf_test_labels_probas: np.array(minibatch_labels_t_probas)} #, \

                        test_predictions, test_loss= session.run([model.test_prediction(), model.loss_test], feed_dict = feed_dict_t)
                        test_pred_bin = get_binary_outputs(test_predictions)

                        session.run(model.acc_test_op, feed_dict = {model.tf_test_labels: np.array(minibatch_labels_t), model.tf_test_labels_pred:  test_pred_bin})
                        session.run(model.recall_test_op, feed_dict = {model.tf_test_labels: np.array(minibatch_labels), model.tf_test_labels_pred: pred_bin})
                        session.run(model.precision_test_op, feed_dict = {model.tf_test_labels: np.array(minibatch_labels), model.tf_test_labels_pred:  pred_bin})

                        accuracy_t = session.run(model.acc_test)
                        recall_t = session.run(model.recall_test)
                        precision_t = session.run(model.precision_test)
                        accuracy = session.run(model.acc)
                        recall = session.run(model.recall)
                        precision = session.run(model.precision)

                        print 'Minibatch training loss at step {0}: {1}'.format(step, l)
                        print 'Minibatch training test at step {0}: {1}'.format(step, test_loss)
                        print 'Current best losses (previous iteration): training: {0}, test: {1}'.format(best_loss, best_test_loss)
                        print "Training set: Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                        print "Test set: Accuracy: %f, Recall: %f, Precision: %f" %(accuracy_t, recall_t, precision_t)

                        print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                        if (test_loss < best_test_loss) and step > 50:
                            for indx in range(len(model.weights)):
                                w = session.run(model.weights[indx])
                                b = session.run(model.biases[indx])
                                model.save_weights(w,identifier+'l_'+str(alpha)+'_best_w'+str(indx))
                                model.save_weights(b,identifier+'l_'+str(alpha)+'_best_b'+str(indx))
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

                            accuracy_accum.append(accuracy)
                            recall_accum.append(recall)
                            precision_accum.append(precision)
                print "Accuracy: %f, Recall: %f, Precision: %f" %(np.mean(accuracy_accum), np.mean(recall_accum), np.mean(precision_accum))

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
