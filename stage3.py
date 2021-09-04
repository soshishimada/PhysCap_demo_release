import sys
import os 
import numpy as np
  
import warnings
import rbdl
import copy 
 
from scipy.spatial.transform import Rotation as Rot  
from Utils.initialize import Initializer
import pybullet as p
from Utils.core_utils import KinematicUtil,angle_util,LabelMotionGetter,Core_utils,RefCorrect
from Utils.util_opt import RbdlOpt
warnings.filterwarnings("ignore") 
 

def sim_loop(path_dict,floor_known=0): 

    ### initializations ###
    AU = angle_util()
    CU = Core_utils()
    kui = KinematicUtil() 
    id_simulator = p.connect(p.DIRECT) 
    p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1) 
     
    model = rbdl.loadModel(path_dict["humanoid_path"].encode())
    model.gravity = np.array([0., -9.81, 0.])
    id_robot = p.loadURDF(path_dict["humanoid_path"], [0, 0, 0.5], globalScaling=1, useFixedBase=False)
    id_robot_vnect = p.loadURDF(path_dict["humanoid_path"], [0, 0, 0.5], globalScaling=1, useFixedBase=False)
    skeleton_specific_base_offset =  np.array([-2.35437, -237.806, 26.4052])
    _, _, jointIds, jointNames = kui.get_jointIds_Names(id_robot)
    LMG = LabelMotionGetter(path_dict["skeleton_filename"],path_dict["motion_filename"], jointNames, skeleton_specific_base_offset)
    la_po_dic = LMG.get_dictionary() 
    if floor_known:
        floor = p.loadURDF(path_dict["floor_path"],  [0, 0.0, 0.0],  [-0.7071068, 0, 0, 0.7071068])
    
    ini = Initializer(floor_known,path_dict["floor_frame"])
    ini.remove_collisions(id_robot,id_robot_vnect) 
    
    rbdl_ids = {"l_ankle": 8, "l_toe": 11, "l_heel": 12, "r_ankle": 17, "r_toe": 21, "r_heel": 22}
    
    contact_flags = np.load(path_dict["contact_path"])
    stationary_flags = np.load(path_dict["stationary_path"])
    
    rc= RefCorrect(stationary_flags)
    rbdl2bullet = ini.get_rbdl2bullet()
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
    l_kafth_ids, r_kafth_ids=ini.get_knee_ankle_foot_toe_heel_ids_rbdl()

    R, T = ini.get_R_T()  
    params = ini.get_params()
    con_j_idx_bullet=ini.get_con_j_idx_bullet()

    ### set up hyperparameters ###
    scale=params["scale"]   
    iter = params["iter"]  
    delta_t =params["delta_t"] 
    j_kp =params["j_kp"]
    j_kd  =params["j_kd"]
    bt_kp  =params["bt_kp"]
    bt_kd =params["bt_kd"]
    br_kp =params["br_kp"]
    br_kd =params["br_kd"]
     
  
    ### RBDL setup ###
    RO = RbdlOpt(delta_t, l_kafth_ids, r_kafth_ids) 
    jointNames = [x.decode('utf-8') for x in jointNames]
    q_all = [] 
    
    floor_height=0.0 
    r_toe_id = con_j_idx_bullet["r_toe_id"] 
    r_heel_id = con_j_idx_bullet["r_heel_id"] 
    l_toe_id =con_j_idx_bullet["l_toe_id"] 
    l_heel_id = con_j_idx_bullet["l_heel_id"] 

    l_toe = np.array(p.getLinkState(id_robot, l_toe_id)[0])
    l_heel = np.array(p.getLinkState(id_robot, l_heel_id)[0])
    r_toe = np.array(p.getLinkState(id_robot, r_toe_id)[0])
    r_heel = np.array(p.getLinkState(id_robot, r_heel_id)[0])
    count=0 

    q = np.zeros(model.q_size)
    qdot = np.zeros(model.qdot_size)
    qddot = np.zeros(model.qdot_size)
    tau = np.zeros(model.qdot_size)
    M = np.zeros((model.q_size, model.q_size))
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
 
    ### start simulation loop ###
    print('=============',la_po_dic.keys())
    n_frames = len(la_po_dic['trans_root_tx']) 
    for count in range(n_frames):
  
        q_ref = LMG.dic2numpy_direct(count + 1, la_po_dic, jointNames) 
        target_com, target_base_ori = LMG.get_base_motion(count + 1, la_po_dic,   trans_scale=0.001)
        target_base_ori_original = copy.copy(target_base_ori)
        target_com += skeleton_specific_base_offset / scale


        ### transform into world corrdinate
        target_com = np.dot(R.T, target_com - (T / scale).reshape(3))  

        corners = CU.get_supp_polygon_corners(model,q,rbdl_ids)
        CoM_projected = CU.get_projected_CoM(model, q, qdot, qddot)
        judgement = CU.support_polygon_checker(CoM_projected, corners)
        kui.motion_update_specification(id_robot_vnect, jointIds, q_ref)
 
        target_base_ori,q_ref=rc.ref_motion_correction(id_robot_vnect,count, target_base_ori, target_base_ori_original, judgement, q, q_ref)
  
        q_ref = q_ref[rbdl2bullet]
        r = Rot.from_euler('zyx', target_base_ori)  
        mat = r.as_matrix() 
        mat =  np.dot(mat, R) 
        r2 = Rot.from_matrix(mat)
        target_vnect_ori = r2.as_euler('xyz')
        target_phy_ori = r2.as_euler('zyx')
 
        q_ref = np.array([target_com[0], target_com[1], target_com[2],  target_phy_ori[0], target_phy_ori[1]  , target_phy_ori[2]] + q_ref.tolist())
         
        if count == 0: 
            p.stepSimulation()
            p.resetBasePositionAndOrientation(id_robot, [target_com[0], target_com[1], target_com[2]], p.getQuaternionFromEuler([target_vnect_ori[0], target_vnect_ori[1] , target_vnect_ori[2]]))
            p.stepSimulation()
 
        for k in range(iter):
            if floor_known:
                """ check mesh collisions """
                bullet_contacts_lth_rth = CU.contact_check(id_robot, contact_flags[count],l_toe_id,l_heel_id,r_toe_id,r_heel_id,floor_height, l_toe,l_heel,r_toe,r_heel) 
            else:
                bullet_contacts_lth_rth = np.zeros(4)

            """ normalize angles """
            q[6:] = np.array(list(map(AU.angle_clean, q[6:])))
            q_ref[3:6] = np.array(list(map(AU.angle_clean, q_ref[3:6])))
            q[3:6] = np.array(list(map(AU.angle_clean, q[3:6])))

            """ set PD controllers """
            torques_root = np.array( [AU.torque_getter(target_rad, current_rad) for target_rad, current_rad in zip(q_ref[3:6], q[3:6])])
            qdot = np.array(qdot)
            pre_qdot = copy.copy(qdot)
            pre_q = copy.copy(q)
            des_qddot = j_kp * (q_ref - q) - j_kd * qdot
            des_qddot[0] = bt_kp  * (q_ref[0] - q[0]) - bt_kd  * qdot[0] 
            des_qddot[1] = bt_kp  * (q_ref[1] - q[1]) - bt_kd  * qdot[1]  
            des_qddot[2] = bt_kp  * (q_ref[2] - q[2]) - bt_kd  * qdot[2] 
            des_qddot[3:6] = br_kp * torques_root - br_kd * qdot[3:6]
            #des_qddot[3:6] = np.clip(des_qddot[3:6],-30,30)
 
            """  get gcc and M """
            gcc = np.zeros(tau.shape)
            rbdl.InverseDynamics(model, q, qdot, np.zeros(des_qddot.shape), gcc)
            rbdl.CompositeRigidBodyAlgorithm(model, q, M, update_kinematics=False)

            """  get Jacobis """
            lth_rth_J6D=CU.get_J_lth_rth(model,q,rbdl_ids)

            """  get foot positions """
            l_toe = np.array(p.getLinkState(id_robot, l_toe_id)[0])
            l_heel = np.array(p.getLinkState(id_robot, l_heel_id)[0])
            r_toe = np.array(p.getLinkState(id_robot, r_toe_id)[0])
            r_heel = np.array(p.getLinkState(id_robot, r_heel_id)[0])

            """  compute GRF and torques """
            GRF_opt, G = RO.qp_force_estimation_toe_heel(bullet_contacts_lth_rth, model, M, q, qdot, des_qddot, gcc, lth_rth_J6D)
            tau, acc, _ = RO.qp_control_hc(bullet_contacts_lth_rth, M, qdot, des_qddot, gcc, lth_rth_J6D,  GRF_opt, G)
            #tau, acc, _ = RO.qp_control_fast(bullet_contacts_lth_rth, M, qdot, des_qddot, gcc, lth_rth_J6D,  GRF_opt, G)
            
            """  Pose update """ 
            q =  pre_q + delta_t * qdot + acc*delta_t * delta_t/2
            """  update visualization """
            r = Rot.from_euler('zyx', q[3:6])
            angle = r.as_euler('xyz')
            if count == 0: q = copy.copy(q_ref)
 
            q_all.append(q)
        
        kui.motion_update_specification(id_robot, jointIds_reordered, q[6:])
        p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], p.getQuaternionFromEuler([angle[2], angle[1], angle[0]]))
        p.stepSimulation()
        p.resetBasePositionAndOrientation(id_robot_vnect, [target_com[0], target_com[1], target_com[2]], p.getQuaternionFromEuler( [target_vnect_ori[2], target_vnect_ori[1], target_vnect_ori[0]]))

        count += 1 
        if count>=(n_frames-1): 
            if not os.path.exists(path_dict["save_path"]):
                os.makedirs(path_dict["save_path"])
            np.save(path_dict["save_path"]+'PhyCap_q.npy',q_all) 
            print("Prediction Saved.") 
            sys.exit() 