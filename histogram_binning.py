import numpy as np
import torch


def histogram_binning_train(pred_in, gt_in, num_bins=15):
    
    preds = []
    labels_onehot = []
    
    if  torch.is_tensor(pred_in):
        pred_in = pred_in.detach().numpy()
    
    if  torch.is_tensor(gt_in):
        gt_in = gt_in.detach().numpy().astype(np.int32)
    
    for pr,gt in zip(pred_in, gt_in):

        n_classes = len(pr)

        one_hot = np.zeros(n_classes)
        one_hot[gt] = 1
        
        preds.append(pr)
        labels_onehot.append(one_hot)
    
    preds = np.array(preds).flatten()
    labels_onehot = np.array(labels_onehot, dtype=int).flatten()
     
    bins = np.linspace(0.0, 1.0, num_bins+1)#[1:]
    bins[0] = bins[0]-0.001
    binned = np.digitize(preds, bins, right=True)-1
        
    bin_accs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for i in range(num_bins):
        bin_sizes[i] = len(preds[binned == i])
        if bin_sizes[i] > 0:
          bin_accs[i] = (labels_onehot[binned==i]).sum() / bin_sizes[i]

    return bin_accs



def histogram_binning_predict(pred_in, bin_accs, num_bins=15):
    
    prediction = np.zeros((pred_in.shape[0]*pred_in.shape[1]))
    
    n_samples = pred_in.shape[0]
    n_classes = pred_in.shape[1]
    
    preds = pred_in.flatten()
    
    bins = np.linspace(0.0, 1.0, num_bins+1)#[1:]
    bins[0] = bins[0]-0.001
    binned = np.digitize(preds, bins, right=True)-1
    
    for i in range(len(preds)):
        Bin = binned[i]
        prediction[i] = bin_accs[Bin]
        
    prediction = prediction.reshape((n_samples,n_classes))    
    return prediction
        
