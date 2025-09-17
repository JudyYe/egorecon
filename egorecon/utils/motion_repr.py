import copy

import numpy as np
import smplx
import torch
from jutils import geom_utils, hand_utils, mesh_utils
from scipy.spatial.transform import Rotation as R


def cano_seq_mano(canoTw, positions, mano_params_dict, return_transf_mat=False, mano_model=None, device='cpu'):
    '''
    Perform canonicalization to the original motion sequence, such that:
    - the sueqnce is z+ axis up
    - frame 0 of the output sequence faces y+ axis  # TODO well our forward change to x+ axis?
    - x/y coordinate of frame 0 is located at origin
    - foot on floor
    Use for AMASS and PROX (coordinate system z axis up)
    input:
        - positions: numpy array, original joint positions (z axis up)
        - smplx_params_dict: dict of numpy array, original smplx params
        - preset_floor_height: if not None, the preset floor height
        - return_transf_mat: if True, also return the transf matrix for canonicalization
    Output:
        - cano_positions: canonicalized joint positions
        - cano_smplx_params_dict: canonicalized smplx params
        - transf_matrix (if return_transf_mat): the transf matrix for canonicalization
    '''
    transf_matrix = canoTw  # (4, 4)
    
    cano_smplx_params_dict, canoJoints = update_globalRT_for_smplx(mano_params_dict, transf_matrix,
                                                      delta_T=None, mano_model=mano_model, device=device)   

    if positions is not None:
        T, J, D = positions.shape  # (T, 21, 3)
        # homogenous coordinates
        wPositions = np.concatenate((positions, np.ones((T, J, 1))), axis=-1) # (T, 21, 4)
        canoPositions = np.matmul(wPositions, transf_matrix.T) # (T, 21, 4）
    else:
        cano_smplx_params_dict_torch = {}
        for key in cano_smplx_params_dict.keys():
            cano_smplx_params_dict_torch[key] = torch.FloatTensor(cano_smplx_params_dict[key])
        canoPositions = mano_model(**cano_smplx_params_dict_torch).joints
    

    if not return_transf_mat:
        return canoPositions, cano_smplx_params_dict
    else:
        return canoPositions, cano_smplx_params_dict, transf_matrix


def update_globalRT_for_smplx(body_param_dict, trans_to_target_origin, mano_model=None, device=None, delta_T=None):
    '''
    input:
        body_param_dict:
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
        delta_T: pelvis location?
        mano_model: my mano wrapper 
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    bs = len(body_param_dict['transl'])

    joints = None
    if delta_T is None:
        body_param_dict_torch = {}
        for key in body_param_dict.keys():
            body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key]).to(device)
        body_param_dict_torch['transl'] = torch.zeros([bs, 3], dtype=torch.float32).to(device)

        smpl_out = mano_model(**body_param_dict_torch)
        # delta_T = smpl_out.joints[:,0,:] # (bs, 3,)
        delta_T = smpl_out.joints[:,0,:] # (bs, 3,)
        delta_T = delta_T.detach().cpu().numpy()
        joints = smpl_out.joints

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_param_dict['global_orient']
    body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [bs, 3,3] rotation mat
    body_T = body_param_dict['transl']
    body_mat = np.zeros([bs, 4, 4])
    body_mat[:, :-1,:-1] = body_R_mat
    body_mat[:, :-1, -1] = body_T + delta_T
    body_mat[:, -1, -1] = 1

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_param_dict)
    trans_to_target_origin = np.expand_dims(trans_to_target_origin, axis=0)  # [1, 4]
    trans_to_target_origin = np.repeat(trans_to_target_origin, bs, axis=0)  # [bs, 4]

    body_mat_new = np.matmul(trans_to_target_origin, body_mat)  # [bs, 4, 4]
    body_R_new = R.from_matrix(body_mat_new[:, :-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:, :-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(-1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(-1,3)
    return body_params_dict_new, joints


def test():

    left_model = smplx.create(
        '/move/u/yufeiy2/pretrain/body_models/mano_v1_2/models/MANO_LEFT.pkl',
        'mano',
        is_rhand=False,
        use_pca=False,
    ).cpu()
    right_model = smplx.create(
        '/move/u/yufeiy2/pretrain/body_models/mano_v1_2/models/MANO_RIGHT.pkl',
        'mano',
        is_rhand=True,
        use_pca=False
    ).cpu()
    model = right_model

    # model_wrapper = {
    #     'left': hand_utils.ManopthWrapper('/move/u/yufeiy2/pretrain/body_models/mano_v1_2/models', side='left'),
    #     'right': hand_utils.ManopthWrapper('/move/u/yufeiy2/pretrain/body_models/mano_v1_2/models', side='right'),
    # }

    T = 4
    body_param_dict = {
        'transl': np.random.randn(T, 3),
        'global_orient': np.random.randn(T, 3),
        'betas': np.random.randn(T, 10),
        'hand_pose': np.random.randn(T, 45),
    }
    body_param_dict = {
        'transl': np.zeros((T, 3)),
        'global_orient': np.zeros((T, 3)),
        'betas': np.zeros((T, 10)),
        'hand_pose': np.zeros((T, 45)),
    }

    body_param_dict_torch = {}
    for key in body_param_dict.keys():
        body_param_dict_torch[key] = torch.FloatTensor(body_param_dict[key])


    canoTw_rot = geom_utils.random_rotations(1,) 
    canoTw_trans = torch.randn(1, 3)
    canoTw = geom_utils.rt_to_homo(canoTw_rot, canoTw_trans)[0]  # (1, 4, 4)
    # print(canoTw.shape, canoTw_rot.shape, canoTw_trans.shape)

    smpl_out = model(**body_param_dict_torch)
    wPositions = smpl_out.joints  # (T, 21, 3)
    print('positions shape: ', wPositions.shape, wPositions)

    canoPositions, cano_smplx_params_dict = cano_seq_mano(canoTw, wPositions, body_param_dict, mano_model=model, device='cpu')
    cano_smplx_params_dict_torch = {}
    for key in cano_smplx_params_dict.keys():
        cano_smplx_params_dict_torch[key] = torch.FloatTensor(cano_smplx_params_dict[key])
    smpl_out_cano = model(**cano_smplx_params_dict_torch)
    canPositions_pred = smpl_out_cano.joints

    # canoPositions = mesh_utils.apply_transform(wPositions, canoTw[None])

    # print(canoPositions - canPositions_pred)
    print(canoPositions.shape, canPositions_pred.shape)
    print(canPositions_pred - canoPositions[..., :3])
    
    return 


if __name__ == '__main__':
    test()