from datetime import datetime
time = datetime.now().strftime('-%m%d%H%M-')

data_root_path_local, save_root_path_local = '/media/wyk/DATA/Datasets/AVE', '/media/wyk/DATA/Results/AVE'
data_root_path_remote, save_root_path_remote = '/home2/wyk/datasets/AVE', '/home2/wyk/results/AVE'

sampling_method = 'uniform'  # uniform, adversarial, harmonized
score_method = 'norm'  # norm, concat, cos
att_method = 'add'  # add, cos
dis_loss = 'hinge'  # hinge, ratio