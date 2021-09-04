import os
from stage3 import sim_loop
from models.networks import ContactEstimationNetwork
import argparse
from stage2 import inferenceCon
import torch
if __name__ == '__main__':

    ### config for fitting and contact calculations ###
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--contact_estimation', type=int, default=0)
    parser.add_argument('--image_size',type=int, default=1024) 
    parser.add_argument('--floor_known', type=int, default=1)
    parser.add_argument('--model_path', default="models/ConStaNet_sample.pkl") 
    parser.add_argument('--floor_frame',  default="data/floor_frame.npy") 
    parser.add_argument('--vnect_2d_path', default="data/sample_vnect_2ds.npy") 
    parser.add_argument('--humanoid_path', default='asset/physcap.urdf') 
    parser.add_argument('--skeleton_filename', default="asset/physcap.skeleton" )
    parser.add_argument('--motion_filename', default="data/sample.motion")
    parser.add_argument('--floor_path', default="asset/plane.urdf") 
    parser.add_argument('--contact_path', default="data/sample_contacts.npy") 
    parser.add_argument('--stationary_path', default="data/sample_stationary.npy")
    parser.add_argument('--save_path', default='./results/')
    args = parser.parse_args()

    ### Contact and Stationary Estimation ###
    if args.contact_estimation:
        target_joints = ["head", "neck", "left_hip",  "left_knee", "left_ankle", "left_toe",  "right_hip", "right_knee", "right_ankle", "right_toe",  "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
        vnect_dic = {"base": 14, "head": 0, "neck": 1, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "left_toe": 16,"right_hip": 8,  "right_knee": 9, "right_ankle": 10, "right_toe": 15, "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,  "right_shoulder": 2, "right_elbow": 3, "right_wrist": 4 }
        window_size=10  
        ConNet = ContactEstimationNetwork(in_channels=32, num_features=512, out_channels=5, num_blocks=4).cuda() 
        ConNet.load_state_dict(torch.load(args.model_path))
        ConNet.eval()
        print("Stage II running ... ")
        inferenceCon(target_joints,vnect_dic,ConNet,args.image_size,window_size,args.vnect_2d_path,args.save_path)
        print("Done. Predictions were saved at "+args.save_path)

    ### Physics-based Optimization ###
    path_dict={ 
            "floor_frame":args.floor_frame, 
            "humanoid_path":args.humanoid_path,
            "skeleton_filename":args.skeleton_filename,
            "motion_filename":args.motion_filename,
            "floor_path":args.floor_path,
            "contact_path":args.contact_path,
            "stationary_path":args.stationary_path,
            "save_path":args.save_path} 
    print("Stage III running ... ") 
    sim_loop(path_dict,floor_known=args.floor_known)
    print("Done. Predictions were saved at "+args.save_path)
    
