# Network Parameters
#n_hidden_1 = 256 # 1st layer number of neurons
#n_hidden_2 = 128 # 2nd layer number of neurons
#n_hidden_3 = 64 # 3th layer number of neurons
#n_hidden_4 = 32 # 4th layer number of neurons
hidden_layers = [128,128,64]
num_input = 8 # inputs  for features
num_classes = 2 # binary classification

LEARNING_RATE = 0.000005
num_steps = 20000
batch_size = 128
display_step = 1000

dataset_path = 'dataset07.2.18.csv'
identifier = '_20.3.18_2layers_'
label_column = 9 # 10th
start_f = 1
end_f = 9
