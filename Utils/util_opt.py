import numpy as np
import sys
import math
sys.path.append("/HPS/Shimada/work/rbdl37/rbdl/build/python")
import rbdl
#import cvxopt
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
class RbdlOpt():
    def __init__(self, delta_t, l_kafth_ids, r_kafth_ids):
        self.delta_t = delta_t
        self.l_kafth_ids = l_kafth_ids
        self.r_kafth_ids = r_kafth_ids

    def c2d_func(self, v):
        vec = np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0],
                        ])
        return vec

    def mat_concatenate(self, mat):
        out = None
        for i in range(len(mat)):
            if i == 0:
                out = mat[i]
            else:
                out = np.concatenate((out, mat[i]), 1)

        return out

    def wrench_separator(self, wrench, contact_info, wrench_dim=6):
        extract_index = [np.arange(i * wrench_dim, (i + 1) * wrench_dim) for i in range(int(len(wrench) / wrench_dim))
                         if contact_info[i]]

        return wrench[np.array(extract_index).reshape(-1)]

    def cross2dot_convert(self, vectors):
        out = np.array(list(map(self.c2d_func, vectors)))
        out = self.mat_concatenate(out)
        return out
    def big_G_getter(self, Gtau):

        G = np.concatenate((Gtau, np.eye(3)), 0)
        G = np.concatenate((G, np.zeros(G.shape)), 1)
        return G

    def get_wrench(self, model, com, q, body_id):
        contact = rbdl.CalcBodyToBaseCoordinates(model, q, body_id, np.zeros(3))
        contact_vec = contact - com
        G_tau_converted = self.cross2dot_convert(np.array([contact_vec]))
        return G_tau_converted
    def jacobi_separator(self, jacobi, contact_info, jacobi_dim=6):

        extract_index = [np.arange(i * jacobi_dim, (i + 1) * jacobi_dim) for i in range(int(len(jacobi) / jacobi_dim))
                         if contact_info[i]]
       
        if len(extract_index) != 0:
            return jacobi[np.array(extract_index).reshape(-1)]
        else:
            return []

    def qp_force_estimation_toe_heel(self, bullet_contacts_lth_rth, model, M, q, qdot, des_qddot, gcc, lr_J6D):
        M = M[:6]
 
        mass = np.zeros(q.shape)
        com = np.zeros(3)
        rbdl.CalcCenterOfMass(model, q, qdot, mass, com)

        l_toe_G_tau_converted = self.get_wrench(model, com, q, self.l_kafth_ids[3])
        l_heel_G_tau_converted = self.get_wrench(model, com, q, self.l_kafth_ids[4])
        r_toe_G_tau_converted = self.get_wrench(model, com, q, self.r_kafth_ids[3])
        r_heel_G_tau_converted = self.get_wrench(model, com, q, self.r_kafth_ids[4])
        R_l_toe = self.big_G_getter(l_toe_G_tau_converted)
        R_l_heel = self.big_G_getter(l_heel_G_tau_converted)
        R_r_toe = self.big_G_getter(r_toe_G_tau_converted)
        R_r_heel = self.big_G_getter(r_heel_G_tau_converted)

        R = np.concatenate((R_l_toe, np.concatenate((R_l_heel, np.concatenate((R_r_toe, R_r_heel), 0)), 0)), 0)

        jacobi = self.jacobi_separator(lr_J6D, bullet_contacts_lth_rth)

        if len(jacobi) == 0:
            return 0, 0

        jacobi = jacobi[:, :6]
 
        R = self.wrench_separator(R, bullet_contacts_lth_rth)
 
        A = np.dot(jacobi.T, R)
 
        b = np.dot(M, des_qddot) + gcc[:6]
 
        W = np.dot(A.T, A)  
 
        Q = -np.dot(b.T, A)  
        mu = 1 / math.sqrt(2)

        G = np.array([[0, 0, -1, 0, 0, 0],
                      [1, 0, -mu, 0, 0, 0],
                      [-1, 0, -mu, 0, 0, 0],
                      [0, 1, -mu, 0, 0, 0],
                      [0, -1, -mu, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1],
                      [0, 0, 0, 1, 0, -mu],
                      [0, 0, 0, -1, 0, -mu],
                      [0, 0, 0, 0, 1, -mu],
                      [0, 0, 0, 0, -1, -mu]
                      ])

        h = np.array(np.zeros(10).tolist())  

        W = matrix(W.astype(np.double))
        Q = matrix(Q.astype(np.double))
        G = matrix(G.astype(np.double))
        h = matrix(h.astype(np.double))

        sol =  solvers.qp(W, Q, G=G, h=h)
        GRF_opt = np.array(sol["x"]).reshape(-1)

        return GRF_opt, R

    def qp_control_hc(self, bullet_contacts_lth_rth, M, qdot, des_qddot, gcc,lr_J6D, GRF_opt, R):
        lr_F_J6D = self.jacobi_separator(lr_J6D, bullet_contacts_lth_rth) 
        if len(lr_F_J6D) != 0:
            general_GRF = np.dot(lr_F_J6D.T, np.dot(R, GRF_opt))
        else:
            general_GRF = 0

        S = np.eye(M.shape[0])
        A = np.concatenate((M, -S), 1)

        b = general_GRF - gcc  # - gcc
        G_top = np.concatenate((-self.delta_t * lr_J6D, np.zeros((lr_J6D.shape[0], M.shape[1]))), 1)
        G_bottom = np.concatenate((self.delta_t * lr_J6D, np.zeros((lr_J6D.shape[0], M.shape[1]))), 1)
        G = np.concatenate((G_top, G_bottom), 0)

        max_vel = 0.01
        max_vel_floor = 0
        max_vel_no_contact = 10000
        l_toe_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact,
                     max_vel_no_contact]
        l_heel_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact,
                      max_vel_no_contact, max_vel_no_contact]
        r_toe_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact,
                     max_vel_no_contact]
        r_heel_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel_no_contact,
                      max_vel_no_contact, max_vel_no_contact]

        if bullet_contacts_lth_rth[0]:
            l_toe_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel, max_vel_floor, max_vel]

        if bullet_contacts_lth_rth[1]:
            l_heel_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel, max_vel_floor, max_vel]

        if bullet_contacts_lth_rth[2]:
            r_toe_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel, max_vel_floor, max_vel]

        if bullet_contacts_lth_rth[3]:
            r_heel_xyz = [max_vel_no_contact, max_vel_no_contact, max_vel_no_contact, max_vel, max_vel_floor, max_vel]

        max_vel_no_contact2 = 10000
        l_toe_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2,
                      max_vel_no_contact2, max_vel_no_contact2]
        l_heel_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2,
                       max_vel_no_contact2, max_vel_no_contact2]
        r_toe_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2,
                      max_vel_no_contact2, max_vel_no_contact2]
        r_heel_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2,
                       max_vel_no_contact2, max_vel_no_contact2]
        max_vel2 = 0.1
        if bullet_contacts_lth_rth[0]:
            l_toe_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel2, max_vel_no_contact2,  max_vel2]

        if bullet_contacts_lth_rth[1]:
            l_heel_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel2, max_vel_no_contact2,  max_vel2]

        if bullet_contacts_lth_rth[2]:
            r_toe_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel2, max_vel_no_contact2,
                          max_vel2]

        if bullet_contacts_lth_rth[3]:
            r_heel_xyz2 = [max_vel_no_contact2, max_vel_no_contact2, max_vel_no_contact2, max_vel2, max_vel_no_contact2,
                           max_vel2] 
        h_top = np.dot(lr_J6D, qdot) + np.array(l_toe_xyz + l_heel_xyz + r_toe_xyz + r_heel_xyz)
        h_bottom = np.array(l_toe_xyz2 + l_heel_xyz2 + r_toe_xyz2 + r_heel_xyz2) - np.dot(lr_J6D, qdot)
        h = np.concatenate((h_top, h_bottom), 0)
        a = np.eye(len(des_qddot))
        bb = des_qddot

        W = np.dot(a.T, a)
        W = np.concatenate((W, np.zeros(M.shape)), 1)
        Q = -np.dot(bb.T, a)
        W_tau_bottom = 0.00001 * np.eye(M.shape[0])
        W_bottom = np.concatenate((np.zeros(M.shape), W_tau_bottom), 1)
        W = np.concatenate((W, W_bottom), 0)

        Q = np.concatenate((Q, np.zeros(des_qddot.shape[0])), 0)
 

        A = matrix(A.astype(np.double))
        b = matrix(b.astype(np.double))
        W = matrix(W.astype(np.double))
        Q = matrix(Q.astype(np.double))
        G = matrix(G.astype(np.double))
        h = matrix(h.astype(np.double))

        sol = solvers.qp(W, Q, A=A, b=b, G=G, h=h)

        x = np.array(sol['x']).reshape(-1)

        tau = x[int(len(x) / 2):].reshape(-1)
        acc = x[:int(len(x) / 2)].reshape(-1)
        return tau, acc, general_GRF


    def qp_control_fast(self, bullet_contacts_lth_rth, M, qdot, des_qddot, gcc,lr_J6D, GRF_opt, R):
        lr_F_J6D = self.jacobi_separator(lr_J6D, bullet_contacts_lth_rth)
 
        if len(lr_F_J6D) != 0:
            general_GRF = np.dot(lr_F_J6D.T, np.dot(R, GRF_opt))
        else:
            general_GRF = 0

        S = np.eye(M.shape[0])
        A = np.concatenate((M, -S), 1)

        b = general_GRF - gcc  # - gcc
 
        a = np.eye(len(des_qddot))
        bb = des_qddot

        W = np.dot(a.T, a)
        W = np.concatenate((W, np.zeros(M.shape)), 1)
        Q = -np.dot(bb.T, a)
        W_tau_bottom = 0.00001 * np.eye(M.shape[0])
        W_bottom = np.concatenate((np.zeros(M.shape), W_tau_bottom), 1)
        W = np.concatenate((W, W_bottom), 0)

        Q = np.concatenate((Q, np.zeros(des_qddot.shape[0])), 0)
 
        A = matrix(A.astype(np.double))
        b = matrix(b.astype(np.double))
        W = matrix(W.astype(np.double))
        Q = matrix(Q.astype(np.double)) 

        sol = solvers.qp(W, Q, A=A, b=b)#, G=G, h=h)

        x = np.array(sol['x']).reshape(-1)

        tau = x[int(len(x) / 2):].reshape(-1)
        acc = x[:int(len(x) / 2)].reshape(-1)
        return tau, acc, general_GRF