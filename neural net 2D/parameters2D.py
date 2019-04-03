# Network Parameters
#hidden_layers = [512,256,256,128]
hidden_layers = [64,64,64,64,64]#64,64,64,64,64] #[64,64,64,64] #,16]
num_input = 23 # inputs  for features
num_classes = 2 # binary classification

LEARNING_RATE = 0.001 # currently using adam so this is just to initialise
num_steps = 20000
batch_size = 256
display_step = 500

pos_weight = 1.0

nNeurons = sum(hidden_layers)

dataset_path = 'automated_11.01.19.csv'
#dataset_path ='dataset_12.12.18_s.csv'
datan = 'dataset_16.12.18_1_sm'
normalL = True
#identifier = 'models2d/28.03_l'+str(len(hidden_layers))+'_w'+str(pos_weight)+'_data_'+datan+'_norm'+ str(normalL) +'_nNeurons'+str(nNeurons)+'noaug/'
identifier = 'models2d/20.03_l'+str(len(hidden_layers))+'_w'+str(pos_weight)+'_data_'+datan+'_norm'+ str(normalL) +'_nNeurons'+str(nNeurons)+'noaug/'
label_column = 23 #-1 # 10th
start_f = 0
end_f = 23

# retraining parameters
#target_data = 'q.csv'
#retrain_steps=20000


target_quad_data = 'test_features_q.csv'
retrain_steps=20000
target_data = 'tri_features.csv'
