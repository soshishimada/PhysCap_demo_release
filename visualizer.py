import pybullet as p
import rbdl
import numpy as np
from Utils.core_utils import KinematicUtil#,angle_util,LabelMotionGetter,Core_utils
from Utils.initialize import Initializer
from scipy.spatial.transform import Rotation as Rot
import time
import argparse

parser = argparse.ArgumentParser(description='arguments for predictions')
parser.add_argument('--q_path',  default="./results/PhyCap_q.npy") 
args = parser.parse_args()
id_simulator = p.connect(p.GUI)
p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0) 

def visualizer(id_robot,q):
    kui.motion_update_specification(id_robot, jointIds_reordered, q[6:]) 
    r = Rot.from_euler('zyx', q[3:6])  # Rot.from_matrix()
    angle = r.as_euler('xyz') 
    p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))
    p.stepSimulation()
    return 0
def data_loader(q_path):
    return np.load(q_path) 
    
 
if __name__ == '__main__': 
    ini = Initializer()
    rbdl2bullet=ini.get_rbdl2bullet() 
    kui = KinematicUtil() 
    qs = data_loader(args.q_path) 
    humanoid_path='./asset/physcap.urdf'
    model = rbdl.loadModel(humanoid_path.encode())
    id_robot = p.loadURDF(humanoid_path, [0, 0, 0.5], globalScaling=1, useFixedBase=False)
     
    _, _, jointIds, _ = kui.get_jointIds_Names(id_robot)
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]  
    for q in qs: 
        visualizer(id_robot,q)  
        time.sleep(0.002)