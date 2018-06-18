# Network Parameters
hidden_layers = [512,256,256,128]
num_input = 13 # inputs  for features
num_classes = 2 # binary classification

LEARNING_RATE = 0.000005 # currently using adam so this is just to initialise
num_steps = 15000
batch_size = 128
display_step = 1000

pos_weight = 1.0

nNeurons = sum(hidden_layers)

dataset_path = '../dataset/dataset_unlimited_0.01_20.4.18.csv'
datan = 'dataset_unlimited_0.01_20.4.18'
normalL = True
identifier = 'models/l'+str(len(hidden_layers))+'_w'+str(pos_weight)+'_data'+datan+'_norm'+ str(normalL) +'_nNeurons'+str(nNeurons)+'/'
label_column = 14 #-1 # 10th
start_f = 1
end_f = 14
