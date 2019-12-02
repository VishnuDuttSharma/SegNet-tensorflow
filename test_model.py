from SegNet import SegNet
from inputs_object import get_filename_list
import cv2
import numpy as np
import os
from tqdm import tqdm

model = SegNet(conf_file='configs/test_config_bs4_ep40k.json')

acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot = 0., 0., 0., 0., 0., None, None
acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot = model.test()

image_filename, label_filename = get_filename_list(model.test_file, model.config)

outdir = './'+model.config['SAVE_MODEL_DIR'].split('/')[1]+'_output/'

os.mkdir(outdir)

for i,fl in enumerate(tqdm(image_filename)):
    fl_nm = fl.split('/')[-1]
    #print(fl_nm)
    cv2.imwrite(outdir+fl_nm, pred_tot[i].reshape((360, 480)).astype('uint8'))
    
 
