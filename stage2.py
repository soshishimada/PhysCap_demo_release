
#from models.networks import ContactEstimationNetwork
import torch 
import numpy as np 
from scipy.ndimage import gaussian_filter1d


def root_relative_2Ds( p_2Ds, bases):
        rr_gt_2Ds = p_2Ds - bases.view(-1, 1, 2) 
        return rr_gt_2Ds

def vnect_smoothing(vnect_2Ds): 
        n_frames = len(vnect_2Ds)
        vnect_2Ds = vnect_2Ds.view(n_frames, -1).numpy().T
        for i in range(len(vnect_2Ds)):
            vnect_2Ds[i] = gaussian_filter1d(vnect_2Ds[i], 2)
        vnect_2Ds = torch.FloatTensor(vnect_2Ds.T).view(n_frames, -1, 2) 
        return vnect_2Ds 
 
def inferenceCon(target_joints,vnect_dic,ConNet,img_size,seqence_len,vnect_file_path,save_path):
    all_con_prediction=[] 
    all_sta_prediction=[] 
    vnect_2Ds = torch.FloatTensor(np.load(vnect_file_path )) 
    vnect2gt = [vnect_dic[key] for key in target_joints]
    vnect_2Ds = vnect_2Ds[:, vnect2gt, :]
    vnect_2Ds = vnect_smoothing(vnect_2Ds)
    vnect_2D_bases = torch.FloatTensor(np.load(vnect_file_path )[:, vnect_dic['base']])
        
    vnect_2Ds /=  img_size
    vnect_2D_bases /= img_size
    vnect_rr_2Ds = root_relative_2Ds(vnect_2Ds, vnect_2D_bases) 
    _,n_j,c = vnect_rr_2Ds.shape
    vnect_rr_2Ds = torch.FloatTensor(vnect_rr_2Ds)

    for i  in range(len(vnect_rr_2Ds)):
 
        if i<seqence_len-1: 
            n_pad = seqence_len-len(vnect_rr_2Ds[:i+1]) 
            pad  = vnect_rr_2Ds[0].view(1,n_j,c).expand(n_pad,-1,-1) 
            seq = torch.cat((pad,vnect_rr_2Ds[:i+1]),0).view(1,seqence_len,n_j,c)
        else:  
            seq = vnect_rr_2Ds[i-seqence_len+1:i+1].view(1,seqence_len,n_j,c)
    
        seq=seq.cuda()   
        pred_labels = ConNet(seq.view(1,seqence_len,-1)) 
        pred_labels = pred_labels.detach().clone().cpu().numpy()
        pred_labels[pred_labels < 0.5] = 0
        pred_labels[pred_labels >= 0.5] = 1 
        all_con_prediction.append(pred_labels[:,:-1])
        all_sta_prediction.append(pred_labels[:,-1]) 

    all_con_prediction=np.array(all_con_prediction)  
    all_sta_prediction=np.array(all_sta_prediction)
    np.save(save_path+'pred_con.npy',all_con_prediction)
    np.save(save_path+'pred_sta.npy',all_sta_prediction)   
    return 0
