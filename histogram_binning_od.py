import numpy as np
from utils_ import divide_detections, text_to_torch
import os 
from os.path import join


def HB_train(labels_folder, preds_folder, num_bins=15, iou_thresh=0.5):
         
    TP, FP, _ = divide_detections(labels_folder, preds_folder, iou_thresh=iou_thresh)
    
    TP = TP[:,0].astype(float)
    FP = FP.astype(float)
    
    bins = np.linspace(0.0, 1.0, num_bins+1)
    bins[0] = bins[0]-0.001

    TP_binned = np.digitize(TP, bins, right=True)-1
    FP_binned = np.digitize(FP, bins, right=True)-1
    
    bin_precs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_sizes[i] = len(TP_binned[TP_binned == i]) + len(FP_binned[FP_binned == i])
        if bin_sizes[i] > 0:
            bin_precs[i] = len(TP_binned[TP_binned == i]) / bin_sizes[i]
            
    return bin_precs




def HB_predict(preds_folder, bin_precs, num_bins=15):
    
    bins = np.linspace(0.0, 1.0, num_bins+1)#[1:]
    bins[0] = bins[0]-0.001
    
    out_folder = "detections_hb"
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    for filename in os.listdir(preds_folder):
        
        preds_file = os.path.join(preds_folder, filename)
        pred_list = text_to_torch(preds_file).tolist()
        
        for n in range(len(pred_list)):  
            for i in range(num_bins):
                
                if bins[i] < pred_list[n][5] <= bins[i+1]:
                    pred_list[n][5] = bin_precs[i]
                    break              
                
        with open(join(out_folder,filename), 'w') as fp:
            for item in pred_list:
                for i in item:
                    fp.write("%s " %i)
                fp.write("\n")