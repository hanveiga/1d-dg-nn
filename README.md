# Shock detection using a Multi layer perceptron in a 1-dimensional discontinuous galerkin code

Work in progress, technical report submitted to ECFD-ECCM and can be found here: 
http://www.eccm-ecfd2018.org/admin/files/filePaper/p1593.pdf


Questions/suggestions/fixes please start a ticket on the repository. 
Enquiries please reach me at hmaria@physik.uzh.ch

Below follows a short description/tutorial of the code.

# Neural net

In folder __neural net__ you'll find the dataset loading, set-up of the neural net and all the training/prediction.

Execute by:

~~~
cd neural\net/
python main.py
~~~

In __parameters.py__ you can configure the architecture and some parameters.

~~~python
hidden_layers = [512,256,256,128] #augmenting this list will add more layers
num_input = 13 # inputs  for features
num_classes = 2 # binary classification
LEARNING_RATE = 0.000005 #size of the gradient jump, this is only necessary for initialisation now that we use Adam algorithm
num_steps = 15000 # number of learning steps
batch_size = 128
display_step = 1000
pos_weight = 1.0 # weight of the positive label for the weighted cross entropy loss function 

dataset_path = '../dataset/dataset_unlimited_0.01_20.4.18.csv'
normalL = True
label_column = 14 #-1 # 10th
start_f = 1
end_f = 14

# this gives the name to the folder containing the model
nNeurons = sum(hidden_layers)
datan = 'dataset_unlimited_0.01_20.4.18'
identifier = 'models/l'+str(len(hidden_layers))+'_w'+str(pos_weight)+'_data'+datan+'_norm'+ str(normalL) +'_nNeurons'+str(nNeurons)+'/'
~~~

# Dataset generation

In folder __solver__ you will find the *CFD model*, this is, a 1-dimensional discontinuous galerkin solver with linear advection and compressible Euler models.

To generate a series of runs and using the high order limiter as a *label maker*, set the integrator parameter in __dg_commons.f90__ to 

~~~fortran
character(LEN=3),parameter::integrator='UNL'
~~~

Then, by doing

~~~
python generate_runs.py
~~~

will run several initial conditions with the shock detection from the high order limiter.

# Integrating model with solver

Once you have a model that you are satisfied with, this will live in 

~~~
neural\net/models/your_model
~~~

do

~~~
cp -r neural\net/models/your_model solver/models/*
~~~

Then in __solver__ folder, edit __ml_parameters.f90__ to point to the right folder and adjust the layer sizes.

To use the limiter, select the parameter

~~~
limiter = 'NN'
~~~

# 2-dimensional version

As the numerical solver codes I am using have not been developed by myself (either as a group effort or my kind advisor gave me a code :-D ), I can't make the numerical codes public, but the neural network related things are online, and the dataset can be found on dropbox:

https://www.dropbox.com/s/qk8lubllro4bey8/automated_11.01.19.csv?dl=0
