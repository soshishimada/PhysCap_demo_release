import math
import pybullet as p
import numpy as np
import copy
import sys
sys.path.append("../")
#sys.path.append("/HPS/Shimada/work/rbdl37/rbdl/build/python")
import rbdl
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
  

class KinematicUtil():
    def motion_update_specification(self, id_robot, jointIds, qs):
        [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
        return 0
    def get_jointIds_Names(self, id_robot):
        jointNamesAll = []
        jointIdsAll = []
        jointNames = []
        jointIds = []
        for j in range(p.getNumJoints(id_robot)):
            info = p.getJointInfo(id_robot, j)
            p.changeDynamics(id_robot, j, linearDamping=0, angularDamping=0)
            jointName = info[1]
            jointType = info[2]
            jointIdsAll.append(j)
            jointNamesAll.append(jointName)
            if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
                jointIds.append(j)
                jointNames.append(jointName)
        return jointIdsAll, jointNamesAll, jointIds, jointNames

class Core_utils():

    def convert_ori_for_vnect(self,target_base_ori):
        R_x_vnect = np.array([[1.0000000, 0.0000000, 0.0000000],
                              [0.0000000, -1.0000000, 0.0000000],
                              [0.0000000, 0.0000000, -1.0000000]])

        r = Rot.from_euler('zyx', copy.copy(target_base_ori))  
        mat = r.as_matrix()
        mat = np.dot(mat, R_x_vnect)  
        r2 = Rot.from_matrix(mat)
        target_vnect_ori_original = r2.as_euler('xyz')
        return target_vnect_ori_original

    def convert_ori_for_vnect2(self,target_base_ori,R):
        R_x_vnect = np.array([[1.0000000, 0.0000000, 0.0000000],
                              [0.0000000, -1.0000000, 0.0000000],
                              [0.0000000, 0.0000000, -1.0000000]])

        r = Rot.from_euler('zyx', copy.copy(target_base_ori)) 
        mat = r.as_matrix()
        mat = np.dot(mat, R)  
        r2 = Rot.from_matrix(mat)
        target_vnect_ori_original = r2.as_euler('xyz')
        return target_vnect_ori_original

    def fcn_RotationFromTwoVectors(self,A, B):
        v = np.cross(A, B)
        v = v / np.linalg.norm(v)
        cos = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
        theta = np.arccos(cos)
        wow = Rot.from_rotvec(1.0 * theta * v)
        R = wow.as_matrix()
        return R

    def rotationMatrixToEulerAngles(self,R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([z, y, x])

    def get_projected_CoM(self,model, q, qdot, qddot):
        CoM = np.zeros(3)
        rbdl.CalcCenterOfMass(model, q, qdot, qddot, CoM)
        CoM_projected = copy.copy(CoM)
        CoM_projected[1] = 0
        return CoM_projected

    def get_J_lth_rth(self,model, q, rbdl_ids):
        l_toe_J6D = np.zeros([6, model.qdot_size])
        l_heel_J6D = np.zeros([6, model.qdot_size])
        r_toe_J6D = np.zeros([6, model.qdot_size])
        r_heel_J6D = np.zeros([6, model.qdot_size])
        rbdl.CalcPointJacobian6D(model, q, rbdl_ids["l_toe"], np.array([0., 0., 0.]), l_toe_J6D)
        rbdl.CalcPointJacobian6D(model, q, rbdl_ids["l_heel"], np.array([0., 0., 0.]), l_heel_J6D)
        rbdl.CalcPointJacobian6D(model, q, rbdl_ids["r_toe"], np.array([0., 0., 0.]), r_toe_J6D)
        rbdl.CalcPointJacobian6D(model, q, rbdl_ids["r_heel"], np.array([0., 0., 0.]), r_heel_J6D)

        lth_J6D = np.concatenate((l_toe_J6D, l_heel_J6D), 0)
        rth_J6D = np.concatenate((r_toe_J6D, r_heel_J6D), 0)
        lth_rth_J6D = np.concatenate((lth_J6D, rth_J6D), 0)

        return lth_rth_J6D

    def get_supp_polygon_corners(self,model, q, rbdl_ids):
        l_toe_coord = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_ids["l_toe"], np.zeros(3))
        l_heel_coord = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_ids["l_heel"], np.zeros(3))
        r_toe_coord = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_ids["r_toe"], np.zeros(3))
        r_heel_coord = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_ids["r_heel"], np.zeros(3))

        corners = np.array([r_toe_coord, l_toe_coord, l_heel_coord, r_heel_coord])
        center_of_supprt = np.average(corners, axis=0)
        return (corners - center_of_supprt) + center_of_supprt
        
    def support_polygon_checker(self,target, corners):
        xz = [0, 2]
        target = target[xz]
        corners = corners[:, xz]
        out = np.array([np.cross(corners[i] - target, corners[0] - corners[i]) if i + 1 == len(corners) else np.cross(
            corners[i] - target, corners[i + 1] - corners[i]) for i in range(len(corners))])
        judgement = np.all(out > 0) if out[0] > 0 else np.all(out < 0)
        return judgement

    def isin_flag(self,target_id, contact_id):
        if target_id in contact_id:
            return 1
        else:
            return 0

    def contact_check(self,id_robot, cflags, l_toe_id, l_heel_id, r_toe_id, r_heel_id, floor_height, l_toe, l_heel, r_toe,r_heel):

        contact_ids = [info[3] for info in p.getContactPoints(bodyA=id_robot)]
        bullet_contacts_lth_rth_raw = [self.isin_flag(l_toe_id, contact_ids), self.isin_flag(l_heel_id, contact_ids), self.isin_flag(r_toe_id, contact_ids), self.isin_flag(r_heel_id, contact_ids)]
         
        bullet_contacts_lth_rth = np.zeros(4)
        bullet_contacts_lth_rth[0] = cflags[0]
        bullet_contacts_lth_rth[1] = cflags[1]
        bullet_contacts_lth_rth[2] = cflags[2]
        bullet_contacts_lth_rth[3] = cflags[3]

        h_thresh = 0.01 + floor_height
         
        if l_toe[1] > h_thresh:  # or l_heel[2] > h_thresh:
            bullet_contacts_lth_rth[0] = 0
            # bullet_contacts_lth_rth[1] = 0
        if l_heel[1] > h_thresh:
            bullet_contacts_lth_rth[1] = 0
        if r_toe[1] > h_thresh:  # or r_heel[2] > h_thresh:
            bullet_contacts_lth_rth[2] = 0
            # bullet_contacts_lth_rth[3] = 0
        if r_heel[1] > h_thresh:
            bullet_contacts_lth_rth[3] = 0
        # bullet_contacts_lth_rth=contact_detection_flags[count-5]+bullet_contacts_lth_rth
        bullet_contacts_lth_rth += bullet_contacts_lth_rth_raw
        
         
        return bullet_contacts_lth_rth

    def torque_getter(self,target_rad, current_rad):
        if target_rad < 0:
            target_rad = math.pi * 2 + target_rad
        if current_rad < 0:
            current_rad = math.pi * 2 + current_rad

        if target_rad >= current_rad:
            A = copy.copy(current_rad)
            B = copy.copy(target_rad)
        else:
            A = copy.copy(target_rad)
            B = copy.copy(current_rad)

        d = min(abs(B - A), abs(math.pi * 2 - B + A))
        # print("d",d)
        if B - A >= math.pi:
            if B == target_rad:
                return -d
            elif B == current_rad:
                return d
            else:
                print("does not match any patterns")
                return 0

        elif B - A < math.pi:
            if B == target_rad:
                return d
            elif B == current_rad:
                return -d
            else:
                print("does not match any patterns")
                return 0
        else:
            print("does not match any patterns")
            return 0


class LabelMotionGetter():
    def __init__(self,skeleton_filename,motion_filename,jointNames,offset):
        self.skeleton_filename = skeleton_filename
        self.motion_filename = motion_filename
        self.jointNames = jointNames
        self.offset = offset  # this is required since the skeleton has some offset from the origin i.e. the root of the skeleton is not located at the origin.
    def dof_name_getter(self):
        with open(self.skeleton_filename) as f:
            content = f.readlines()
        content = np.array([x.strip().split(" ") for x in content])
        dof_names_mess = np.array(content[99:][np.arange(0, len(content[99:]), 3)[:-1]])
        dof_names = [x[0] for x in dof_names_mess]
        return dof_names

    def motion_data_getter(self):
        with open( self.motion_filename) as f:
            content = f.readlines() 
        content = np.array([x.strip().split(" ") for x in content])[1:]

        cleaned_pose = []

        for line in content:
            test = np.array([float(x) for x in line if not x == ""])[1:]
            cleaned_pose.append(test.tolist())
        cleaned_pose = np.array(cleaned_pose).T

        return cleaned_pose
    def dic_create_label_pose(self):
        dic_frames = {}
        for label, data in zip(self.dof_names, self.cleaned_pose):
            dic_frames[label] = data
        for name in self.jointNames:
            if name.decode("utf-8") not in dic_frames.keys():
                dic_frames[name.decode("utf-8")] = np.zeros(len(self.cleaned_pose[0]))
        return dic_frames

    def get_dictionary(self):
        self.dof_names = self.dof_name_getter()
        self.dof_names = [name.replace("rot_","") for name in self.dof_names]
        self.cleaned_pose = self.motion_data_getter()
        return self.dic_create_label_pose()

    def get_base_motion(self,  frame, la_po_dic, trans_scale):
        ref_base_config = self.dic2numpy(frame, la_po_dic, [b'trans_root_tx', b'trans_root_ty', b'trans_root_tz', b'root_rx', b'root_ry', b'root_rz'])
        target_com =  ref_base_config[:3]  * trans_scale  
        target_base_ori = copy.copy(ref_base_config[3:]) 
        return np.array(target_com), np.array(target_base_ori)

    def dic2numpy(self,frame,la_po_dic,jointNames):
        return np.array([la_po_dic[name.decode('utf-8')][frame] for name in jointNames])

    def dic2numpy_direct(self,frame,la_po_dic,jointNames):
        return np.array([la_po_dic[name][frame] for name in jointNames])

class angle_util():
    def angle_clean(self,q):
        mod = q % (2 * math.pi)
        if mod >= math.pi:
            return mod - 2 * math.pi
        else:
            return mod
    def modder(self,radian):
        if radian >= 0:
            return radian%(math.pi*2)
        else:
            return radian%(-math.pi*2)

    def positive_rad(self,radian):
        if radian < 0:
            return radian + math.pi*2
        else:
            return radian

    def get_clean_angle(self,radian):
        #returns angle between 0 and 2pi
        return np.array(list(map(lambda x: self.positive_rad(self.modder(x)), radian)))

    def torque_getter(self,target_rad, current_rad):

        if target_rad >= current_rad:
            A = copy.copy(current_rad)
            B = copy.copy(target_rad)
        else:
            A = copy.copy(target_rad)
            B = copy.copy(current_rad)

        d = min(abs(B - A), abs(math.pi * 2 - B + A))
 
        if B - A >= math.pi:
            if B == target_rad:
                return -d
            elif B == current_rad:
                return d
            else:
                print("does not match any patterns")
                return 0

        elif B - A < math.pi:
            if B == target_rad:
                return d
            elif B == current_rad:
                return -d
            else:
                print("does not match any patterns")
                return 0
        else:
            print("does not match any patterns")
            return 0

    def compute_difference(self,target_rads, current_rads): 
        target_rads = self.get_clean_angle(target_rads)
        current_rads = self.get_clean_angle(current_rads) 
        torques = [self.torque_getter(target_rad, current_rad) for target_rad,current_rad in zip(target_rads, current_rads)]
        if math.pi/2<=current_rads[2] and current_rads[2]<= 3*math.pi/2:
            torques[0] = -torques[0]
        if math.pi / 2 <= current_rads[2] and current_rads[2] <= 3 * math.pi / 2:
            torques[1] = -torques[1] 
        return np.array(torques)

AU = angle_util()
CU = Core_utils()
class RefCorrect():
    def __init__(self,stationary_flags):
        self.knee_correct_count=0
        self.inter_flag=0
        self.inter_count=0
        self.pre_correct_flag=0
        self.stationary_flags=stationary_flags

    def ref_motion_correction(self,id_robot_vnect,count,target_base_ori,target_base_ori_original,judgement,q,q_ref):

        end = p.getLinkState(id_robot_vnect, 47)[0]# 47 top torso
        basePos, _ = p.getBasePositionAndOrientation(id_robot_vnect)
        vec_from = np.array(end) - np.array(basePos)
        vec_to = np.array([0, 1, 0])
        R_correct = CU.fcn_RotationFromTwoVectors(vec_from, vec_to)

        """  this R conversion is necessary due to the difference of euler convention """
        eulerR = CU.rotationMatrixToEulerAngles(R_correct)
        r_calib = Rot.from_euler('xyz', eulerR)
        R_calib = r_calib.as_matrix()

        r3 = Rot.from_euler('zyx', target_base_ori)
        mat = r3.as_matrix()
        mat = np.dot(mat, R_calib)
        r4 = Rot.from_matrix(mat)
        target_vec = r4.as_euler('zyx')

        key_times = [0, 1]
        current_r = Rot.from_euler('zyx', np.array(list(map(AU.angle_clean, target_base_ori_original))))
        xyz_base = current_r.as_euler('zyx')
        r1 = Rot.from_euler('zyx', [xyz_base, target_vec]) # -target_base_ori[1]+
        slerp = Slerp(key_times, r1)
        times = np.arange(0, 1.0, 0.05)
        interp_rots = slerp(times)
        eulers = interp_rots.as_euler('zyx') 

        if self.stationary_flags[count] and not judgement:
            self.inter_flag = 1

        if not self.stationary_flags[count]:
            self.inter_flag = 0
            self.inter_count = 0
            self.pre_correct_flag = 0

        if self.inter_flag and abs(q[3]) < 0.2:
            # p.addUserDebugText(" im in", [0, 1, 0], [0, 0, 1], textSize=3, replaceItemUniqueId=30)
            target_base_ori = eulers[self.inter_count]
            if abs(target_base_ori[0] - q[0]) > 2.7 or abs(target_base_ori[2] - q[2]) > 2.7:
                target_base_ori[0] += math.pi
                target_base_ori[1] = math.pi - target_base_ori[1]
                target_base_ori[2] += math.pi
            if self.inter_count != len(eulers) - 1 and not judgement:
                self.inter_count += 1
            if abs(q[3]) < 0.1 and not judgement:
               # p.addUserDebugText(str(self.knee_correct_count), [0, 1.1, 0], [1, 0, 0], textSize=3, replaceItemUniqueId=31)
                w = 0.01
                q_ref[3] /= w * self.knee_correct_count + 1
                q_ref[4] /= w * self.knee_correct_count + 1
                q_ref[2] /= w * self.knee_correct_count + 1
                q_ref[1] /= w * self.knee_correct_count + 1
                q_ref[14] /= w * self.knee_correct_count + 1
                q_ref[13] /= w * self.knee_correct_count + 1
                q_ref[12] /= w * self.knee_correct_count + 1
                q_ref[11] /= w * self.knee_correct_count + 1
                self.knee_correct_count += 1
            else:
                self.knee_correct_count = 0
                #p.addUserDebugText(" ", [0, 1, 0], [0, 0, 1], textSize=3, replaceItemUniqueId=30)
        else:
            self.knee_correct_count = 0 
    
        return target_base_ori,q_ref