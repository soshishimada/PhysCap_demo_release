import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse



def vnect_smoothing(vnect_2Ds): 
        n_frames = len(vnect_2Ds)
        vnect_2Ds = vnect_2Ds.reshape(n_frames, -1).T
        for i in range(len(vnect_2Ds)):
            vnect_2Ds[i] = gaussian_filter1d(vnect_2Ds[i], 2)
        vnect_2Ds = (vnect_2Ds.T).reshape(n_frames, -1, 2) 
        return vnect_2Ds

def vnect_mdd_loader(filename):
    with open(filename) as f:
        content = f.readlines() 
    content = np.array([ x.strip().split("," ) for x in content][1:])
    content = np.array([ np.array(list(map(float,x)))  for x in content] )[:,1:].reshape(len(content),-1,2)
    return content
 

def main(input_path,output_folder,smoothing):
    data = vnect_mdd_loader(input_path)
    if smoothing:
        data = vnect_smoothing(data)
    np.save(output_folder+"vnect_2ds.npy",data)
    return 0

if __name__ == '__main__':

    ### config for fitting and contact calculations ###
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--input', type=str, default="./VNect_data/ddd.mdd")
    parser.add_argument('--output', type=str, default="./VNect_data/")
    parser.add_argument('--smoothing', type=int, default=0)
    args = parser.parse_args()
    main(args.input,args.output,args.smoothing)






