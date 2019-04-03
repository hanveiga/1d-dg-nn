import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
import logging
logging.getLogger().setLevel(logging.INFO)
import os

from data2D import load_data
from parameters2D import *
from models_explicit2D import GenericModel

stride = 10 # 5 * 10

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
    best_test_loss = 1000
    old_test_loss = 1000
    incr = 0
    train_losses = np.zeros([10,1])
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        session.run(model.running_vars_initializer)
        session.run(model.running_vars_initializer2)

        nepocs = 10

        for epoch in range(nepocs):
            batch_num = data.train._labels.shape[0]/batch_size
            print(batch_num)
            for step in range(batch_num):
                offset = (step * batch_size) % (data.train._labels.shape[0] - batch_size)
                minibatch_data = data.train._features[offset:(offset + batch_size), :]
                minibatch_labels_probas = data.train._labels_probas[offset:(offset + batch_size)]
                minibatch_labels = data.train._labels[offset:(offset + batch_size)]

                # training dictionary
                feed_dict = {model.tf_train_dataset : minibatch_data, model.tf_train_labels : np.array(minibatch_labels),\
                            model.tf_train_labels_probas: np.array(minibatch_labels_probas)}

                # Start training
                _, l, predictions = session.run([model.optimizer, model.loss, model.train_prediction()], feed_dict = feed_dict)

                # Return loss on training set
                pred_bin = get_binary_outputs(predictions)

                # Fetch parameters to be able to print them
                session.run(model.acc_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})
                session.run(model.recall_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred: pred_bin})
                session.run(model.precision_op, feed_dict = {model.tf_train_labels: np.array(minibatch_labels), model.tf_train_labels_pred:  pred_bin})

                accuracy = session.run(model.acc)
                recall = session.run(model.recall)
                precision = session.run(model.precision)

                if step % 10 == 0:
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

                    if test_loss > old_test_loss:
                        incr +=1
                    else:
                        incr = 0
                    old_test_loss = test_loss


                # computing pk
                if step % display_step == 0:
                    print 'Minibatch training loss at step {0}: {1}'.format(step, l)
                    print 'Minibatch training test at step {0}: {1}'.format(step, test_loss)
                    print 'Current best losses (previous iteration): training: {0}, test: {1}'.format(best_loss, best_test_loss)
                    print "Training set: Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                    print "Test set: Accuracy: %f, Recall: %f, Precision: %f" %(accuracy_t, recall_t, precision_t)
                    if (best_test_loss > test_loss) and (best_test_loss<10) and (l<10):
                        print 'saving model...'
                        print '@ losses: training: {0}, test: {1}'.format(l, test_loss)
                        for indx in range(len(model.weights)):
                            w = session.run(model.weights[indx])
                            b = session.run(model.biases[indx])
                            model.save_weights(w,identifier+'best_w'+str(indx))
                            model.save_weights(b,identifier+'best_b'+str(indx))

                    best_loss = np.min([best_loss,l])
                    best_test_loss = np.min([best_test_loss,test_loss])
                print incr
                if epoch > 0:
                    if incr > 5:
                        print 'we are overfitting'
                        # overfitting
                        return
                    # if we passed though the whole dataset once, we can consider early stop
                    # criteria is the test error keeps increasing (for 100 iterations, for example...)



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
                #with open(identifier+"accuracy.txt","a") as text_file:
                #    text_file.write("Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision))

                #print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
                accuracy_accum.append(accuracy)
                recall_accum.append(recall)
                precision_accum.append(precision)

            print('Validation: Mean Accuracy: %f, Mean Recall: %f, Mean Precision: %f' %(np.mean(accuracy_accum), np.mean(recall_accum), np.mean(precision_accum)))
            with open(identifier+"accuracy.txt","a") as text_file:
                text_file.write('Mean Accuracy: %f, Mean Recall: %f, Mean Precision: %f \n' %(np.mean(accuracy_accum), np.mean(recall_accum), np.mean(precision_accum)))


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
    #print "Accuracy: %f, Recall: %f, Precision: %f" %(accuracy, recall, precision)
    return accuracy, recall, precision

if __name__=='__main__':
    main()
