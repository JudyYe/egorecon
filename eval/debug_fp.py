
import pickle
import os.path as osp
from eval.eval_utils import eval_pose
from jutils import mesh_utils, image_utils
from pytorch3d.structures import Meshes
import torch
import numpy as np
from egorecon.visualization.pt3d_visualizer import render_scene_from_azel, Pt3dVisualizer
from jutils import geom_utils, plot_utils


def vis(gt_wTo, pred_wTo, oMesh, save_file="outputs/debug/vis_pose_metrics"):
    """
    Visualize ground truth and predicted object meshes with per-frame metrics.
    
    :param gt_wTo: (T, 4, 4) ground truth world-to-object transforms
    :param pred_wTo: (T, 4, 4) predicted world-to-object transforms
    :param oMesh: trimesh.Trimesh or PyTorch3D Meshes object
    :param save_file: Path to save the output video
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert inputs to tensors if needed
    if isinstance(gt_wTo, np.ndarray):
        gt_wTo = torch.FloatTensor(gt_wTo).to(device)
    if isinstance(pred_wTo, np.ndarray):
        pred_wTo = torch.FloatTensor(pred_wTo).to(device)
    
    T = gt_wTo.shape[0]
    
    # Convert trimesh to PyTorch3D Meshes if needed
    if hasattr(oMesh, 'vertices') and hasattr(oMesh, 'faces'):
        # It's a trimesh object
        o_verts = torch.FloatTensor(oMesh.vertices).to(device)[None]  # (1, V, 3)
        o_faces = torch.LongTensor(oMesh.faces).to(device)[None]  # (1, F, 3)
        oMesh_pt3d = Meshes(verts=o_verts, faces=o_faces).to(device)
    elif isinstance(oMesh, Meshes):
        # Already a PyTorch3D Meshes
        oMesh_pt3d = oMesh.to(device)
    else:
        raise ValueError(f"Unsupported mesh type: {type(oMesh)}")
    
    # Get mesh vertices and faces
    o_verts = oMesh_pt3d.verts_padded()  # (1, V, 3)
    o_faces = oMesh_pt3d.faces_padded()  # (1, F, 3)
    
    # Transform meshes using gt_wTo and pred_wTo
    # Expand o_verts to (T, V, 3)
    o_verts_expanded = o_verts.repeat(T, 1, 1)  # (T, V, 3)
    o_faces_expanded = o_faces.repeat(T, 1, 1)  # (T, F, 3)
    
    # Apply transforms
    wMesh_gt_verts = mesh_utils.apply_transform(o_verts_expanded, gt_wTo)  # (T, V, 3)
    wMesh_pred_verts = mesh_utils.apply_transform(o_verts_expanded, pred_wTo)  # (T, V, 3)
    
    # Create Meshes objects with colors
    wMesh_gt = Meshes(verts=wMesh_gt_verts, faces=o_faces_expanded).to(device)
    wMesh_gt.textures = mesh_utils.pad_texture(wMesh_gt, 'red')
    
    wMesh_pred = Meshes(verts=wMesh_pred_verts, faces=o_faces_expanded).to(device)
    wMesh_pred.textures = mesh_utils.pad_texture(wMesh_pred, 'blue')
    
    wScene = mesh_utils.join_scene([wMesh_gt, wMesh_pred])

    # Get per-frame metrics using eval_pose
    metrics = eval_pose(
        gt_wTo,
        pred_wTo,
        oMesh_pt3d,
        metric_names=["add", "adi", "center"]
    )
    
    # Create text list for each frame with metrics
    text_list = []
    for t in range(T):
        text_one_frame = ""
        for metric_name in ["add", "adi", "center"]:
            if metric_name in metrics:
                value = metrics[metric_name][t]
                text_one_frame += f"{metric_name}: {value:.4f}\n"
        text_list.append([text_one_frame])
    
    # Render scene
    # Get scene bounds for normalization
    all_verts = torch.cat([wMesh_gt.verts_packed(), wMesh_pred.verts_packed()], dim=-2)  # (T, 2V, 3)
    nTw = mesh_utils.get_nTw(all_verts[None], new_scale=1.2)  # (T, 4, 4)
    

    coord = plot_utils.create_coord(device, len(wScene), size=0.2)
    wScene = mesh_utils.join_scene([wScene, coord])
    # Render each frame
    image = render_scene_from_azel(wScene, nTw, az=160, el=5, out_size=360)  # (T, 3, H, W)
    image_list = image["image"]
    # Stack images: (T, 3, H, W)
    image_tensor = image_list.unsqueeze(1)
    print("image_tensor", image_tensor.shape)
    
    # Add text overlay using save_gif
    image_utils.save_gif(
        image_tensor,  # (T, 1, 3, H, W)
        save_file,
        text_list=text_list,
        fps=30,
        ext=".mp4"
    )
    
    print(f"Saved visualization to {save_file}.mp4")
    return image_tensor, text_list

def load_one(index="003132_000020"):
    object_mesh_dir = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/object_models_eval"
    object_library = Pt3dVisualizer.setup_template(object_mesh_dir)
    pred_dir = "outputs/first_frame_hoi/firstFalse_dynTrue_static100_contact_100_smoothness10/eval_hoi_contact_ddim_long_vis/post/"
    pred_file = osp.join(pred_dir, f"{index}.pkl")

    gt_file = "/move/u/yufeiy2/egorecon/data/HOT3D-CLIP/preprocess/dataset_contact.pkl"
    with open(gt_file, "rb") as f:
        gt_data = pickle.load(f)
    seq, obj = index.split("_")
    gt_wTo = gt_data[seq][f"obj_{obj}_wTo"]
    
    with open(pred_file, "rb") as f:
        pred_data = pickle.load(f)
    pred_wTo = torch.FloatTensor(pred_data["wTo"])
    pred_wpTo = geom_utils.se3_to_matrix_v2(pred_wTo)[0]
    T = pred_wTo.shape[0]


    # align by the 1st frame camera
    predwTc = torch.FloatTensor(pred_data["wTc"])[0]
    wpTc = geom_utils.se3_to_matrix_v2(predwTc) # (4, 4)
    gtwTc = torch.FloatTensor(gt_data[seq]["wTc"])[0] # (4, 4) 
    wTwp = gtwTc @ wpTc.inverse()
    pred_wTo = wTwp[None].repeat(T, 1, 1) @ pred_wpTo
    pred_wTo = pred_wTo.cpu().numpy()   
    
    oMesh = object_library[obj]
    print(gt_wTo.shape, pred_wTo.shape, )
    vis(gt_wTo, pred_wTo, oMesh, save_file=f"outputs/debug/vis_pose_metrics_{index}.mp4")
    
if __name__ == "__main__":
    load_one()