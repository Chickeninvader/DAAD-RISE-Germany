import pandas as pd

batch_size = 16
loss = 'BCE'
num_epochs = 10
lr = 0.0001
model_name = 'resnet_3d'
additional_saving_info = ''
pretrained_path = None
save_file = False
representation = 'gaussian'

metadata = pd.read_excel('critical_classification/dashcam_video/metadata.xlsx')
