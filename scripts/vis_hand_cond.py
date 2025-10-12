import numpy as np
from scipy.spatial.transform import Rotation
import rerun as rr
import torch
import pickle
from jutils import geom_utils, mesh_utils, plot_utils, image_utils
from pytorch3d.structures import Meshes
from egorecon.manip.model.guidance_optimizer_jax import project_jax

def vis_hand_cond():
    with open("outputs/debug/hand.pkl", 'rb') as f:
        data = pickle.load(f)

    wHands = data["wHands"]  # (B, T, J, 3) 
    B, T, J_3 = wHands.shape
    J = J_3 // 3

    coord = data["hand"].reshape(B, T, J, 3)  # (B, T, J, 3)
    wHands = wHands.reshape(B, T, J, 3)
    wTo = data["wTo"]  # (B, T, 3+6)
    oPoints = data["oPoints"]  # (B, P, 3)

    max_T = 120
    for  b in range(B):
        vis_hand_cond_one_rr(wHands[b][:max_T], wTo[b][:max_T], oPoints[b], coord[b][:max_T], pref=f"b_{b}")
        if b >= 2:
            break

def vis_hand_cond_one_rr(wHands, wTo, oPoints, coord, pref):
    """use rerun 

    :param wHands: _description_
    :param wTo: _description_
    :param oPoints: _description_
    :param coord: _description_
    :param pref: _description_
    """
    T = wHands.shape[0]
    J = wHands.shape[1]
    P = oPoints.shape[0]
    wNN = coord + wHands

    wTo = geom_utils.se3_to_matrix_v2(wTo)  # (T, 4, 4)
    wHands = wHands.cpu().detach().numpy()
    wTo = wTo.cpu().detach().numpy()
    oPoints = oPoints.cpu().detach().numpy()
    coord = coord.cpu().detach().numpy()
    wNN = wNN.cpu().detach().numpy()

    rr.init(f"outputs/debug/vis_hand_cond_{pref}")
    rr.save(f"outputs/debug/vis_hand_cond_{pref}.rrd")
    for t in range(T):
        rr.set_time_sequence("frame", t)

        if t == 0:
            rr.log("world/object", rr.Points3D(
                positions=oPoints,
            ))
        rr.log("world/object", rr.Transform3D(
            translation=wTo[t, :3, 3],
            rotation=rr.Quaternion(xyzw=Rotation.from_matrix(wTo[t, :3, :3]).as_quat())
        ))

        rr.log("world/hand", rr.Points3D(
            positions=wHands[t],
        ))

        # log line segment from wNN to wHands  
        print('wNN', wNN[t].shape, 'wHands', wHands[t].shape)  # (J, 3)
        line = np.stack([wNN[t], wHands[t]], axis=1)  # (J, 2, 3)
        rr.log("world/line", rr.LineStrips3D(
            line,
            colors=[[0, 0, 255]],
            radii=[0.01],
        ))

    return 
def vis_hand_cond_one(wHands, wTo, oPoints, coord, pref):
    T = wHands.shape[0]  # (T, J, 3)
    J = wHands.shape[1]  # (T, J, 3)
    P = oPoints.shape[0]  # (P, 3)
    wNN = coord + wHands  # (T, J, 3)
    
    print(wHands.shape, wTo.shape, oPoints.shape, coord.shape)
    print(wHands.device, wTo.device, oPoints.device, coord.device)

    wTo = geom_utils.se3_to_matrix_v2(wTo)  # (B, T, 4, 4)
    wTo = wTo.reshape(T, 4, 4)
    oPoints = oPoints[None].repeat(T, 1, 1)
    wPoints = mesh_utils.apply_transform(oPoints, wTo)  # (B*T, P, 3)

    wPoints_mesh = plot_utils.pc_to_cubic_meshes(wPoints, )
    wPoints_mesh.textures = mesh_utils.pad_texture(wPoints_mesh, 'white')

    wHands_mesh = plot_utils.pc_to_cubic_meshes(wHands.reshape(T, J, 3), eps=0.02)
    wHands_mesh.textures = mesh_utils.pad_texture(wHands_mesh, 'blue')

    verts, faces = plot_utils.create_line(wNN.reshape(T*J, 3),wHands.reshape(T*J, 3))
    verts = verts.reshape(T, -1, 3)
    faces = faces.reshape(T, -1, 3)
    wLines = Meshes(verts=verts, faces=faces).cuda()
    wLines.textures = mesh_utils.pad_texture(wLines, 'red')

    wScene = mesh_utils.join_scene([wPoints_mesh, wHands_mesh, wLines])
    wScene = wScene.cuda()
    print(wScene.device)
    image_list = mesh_utils.render_geom_rot_v2(wScene, )
    image_list = torch.stack(image_list, axis=0)
    time_len, B, H, W, C = image_list.shape
    print('image_list', image_list.shape)
    image_utils.save_gif(image_list.reshape(time_len * B, 1, H, W, C), f"outputs/debug/vis_hand_cond_{pref}", ext='.mp4')
    



if __name__ == "__main__":
    vis_hand_cond()