import numpy as np
import pybullet as p
class Initializer():
    def __init__(self,floor_known=None,floor_frame_path=None,): 
        if floor_known:
            self.RT = np.load(floor_frame_path)
        else:
            self.RT =np.eye(4)
            
        self.rbdl2bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 20, 21, 22]

        r_knee_id = 26
        r_ankle_id = 28
        r_foot_id = 30
        r_toe_id = 34
        r_heel_id = 35

        self.r_kafth_ids = [r_knee_id, r_ankle_id, r_foot_id, r_toe_id, r_heel_id]
 
        l_knee_id = 9
        l_ankle_id = 11
        l_foot_id = 13
        l_toe_id = 17
        l_heel_id = 18
        self.l_kafth_ids = [l_knee_id, l_ankle_id, l_foot_id, l_toe_id, l_heel_id]

        self.params1={
            "scale":1000,"iter":8,"delta_t":0.001,"j_kp":117497,"j_kd":3300,"bt_kp":155000,
            "bt_kd":2300,"br_kp":50000,"br_kd":2800}
        self.params2={
            "scale":1000,"iter":8,"delta_t":0.01,"j_kp":300,"j_kd":150,"bt_kp":600,
            "bt_kd":300,"br_kp":300,"br_kd":150}
        self.con_j_ids_bullet = {"r_toe_id":34,"r_heel_id":35,"l_toe_id":17,"l_heel_id":18}
    
    def get_params(self):
        return self.params2#self.params1

    def get_con_j_idx_bullet(self):
        return self.con_j_ids_bullet

    def remove_collisions(self,id_a,id_b):

        ### turn of collision between humanoids ###
        for i in range(p.getNumJoints(id_a)):
            for j in range(p.getNumJoints(id_b)):
                p.setCollisionFilterPair(id_a, id_b, i, j, 0)
        return 0

    def get_knee_ankle_foot_toe_heel_ids_rbdl(self):
        return self.l_kafth_ids,self.r_kafth_ids

    def get_rbdl2bullet(self):
        return self.rbdl2bullet
    def change_humanoid_color(self,id_robot,color):
        for j in range(p.getNumJoints(id_robot)):
            p.changeVisualShape(id_robot, j, rgbaColor=color)
        return 0

    def get_R_T(self): 
        R = self.RT[:3, :3]
        T = self.RT[:-1, 3:].reshape(3)
        return R,T
