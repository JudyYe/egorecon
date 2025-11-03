# run foundation pose on HOT3D-CLIP
import os.path as osp

import numpy as np
import torch
from jutils import geom_utils, hand_utils, image_utils, mesh_utils
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shader import HardFlatShader
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures import Meshes

from egorecon.manip.data.utils import load_pickle
from egorecon.utils.motion_repr import HandWrapper
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer

mano_model_folder = "assets/mano"

seq_name = "002240"
all_dat = load_pickle("data/HOT3D-CLIP/preprocess/dataset_contact.pkl")
device = torch.device("cuda:0")


vis_dir = "outputs/debug_fp_mask"


def discretize_rgb_to_obj_uid(rgb, index, index2objid, fg_mask):
    """
    Batched discretization: map RGB image to closest object texture index using torch.
    Background pixels (fg_mask == 0) are not assigned to any object.

    Args:
        rgb: (N, 3, H, W) tensor, RGB image [0, 1]
        index: numpy array of object indices (uint8 values)
        index2objid: dict mapping index -> obj_id
        fg_mask: (N, 1, H, W) or (N, H, W) tensor, foreground mask (1=foreground, 0=background)

    Returns:
        obj_mask: (N, #obj, H, W) torch tensor, binary mask per object
    """
    device = rgb.device
    num_obj = len(index)

    # Normalize fg_mask to (N, H, W)
    if fg_mask.dim() == 4:
        fg_mask = fg_mask[:, 0, :, :]  # (N, H, W)
    fg_mask = fg_mask.squeeze().float()  # (N, H, W)

    # Extract R channel and convert to uint8 range, keep as torch tensor
    r_channel = rgb[:, 0, :, :] * 255.0  # (N, H, W), [0, 255]

    # Convert index to torch tensor
    index_tensor = torch.from_numpy(index.astype(np.float32)).to(device)  # (num_obj,)

    # Broadcast: (N, H, W, 1) - (1, 1, 1, num_obj) = (N, H, W, num_obj)
    r_expanded = r_channel.unsqueeze(-1)  # (N, H, W, 1)
    index_expanded = (
        index_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, 1, num_obj)

    # Find closest index for all pixels at once
    distances = torch.abs(r_expanded - index_expanded)  # (N, H, W, num_obj)
    closest_idx_pos = torch.argmin(distances, dim=-1)  # (N, H, W)

    # Create one-hot encoding: (N, num_obj, H, W)
    N, H, W = closest_idx_pos.shape
    obj_mask = torch.nn.functional.one_hot(
        closest_idx_pos, num_classes=num_obj
    ).float()  # (N, H, W, num_obj)
    obj_mask = obj_mask.permute(0, 3, 1, 2)  # (N, num_obj, H, W)

    # Mask out background pixels: set all channels to 0 for background
    fg_mask_expanded = fg_mask.unsqueeze(1)  # (N, 1, H, W)
    obj_mask = obj_mask * fg_mask_expanded  # (N, num_obj, H, W)

    return obj_mask


def get_mask(seq_data, object_library, sided_model: HandWrapper):
    # mask:
    # {'uid': [T, H, W]}
    # create scene
    wScene = []

    index = (np.linspace(0.1, 0.9, len(seq_data["objects"])) * 255).astype(np.uint8)
    objid2index = {}
    for i, obj_id in enumerate(seq_data["objects"]):
        objid2index[obj_id] = index[i]
    index2objid = {index: obj_id for obj_id, index in objid2index.items()}
    print(index)

    for i, (obj_id, obj_data) in enumerate(seq_data["objects"].items()):
        wTo = torch.FloatTensor(obj_data["wTo"]).to(device)
        oObj = object_library[obj_id].to(device)
        T = len(wTo)
        oObj_exp = Meshes(
            verts=oObj.verts_padded().repeat(T, 1, 1),
            faces=oObj.faces_padded().repeat(T, 1, 1),
        ).to(device)
        wObj = mesh_utils.apply_transform(oObj_exp, wTo)
        color = torch.zeros_like(wObj.verts_padded())
        color[..., 0] += 1.0 * index[i] / 255
        wObj.textures = mesh_utils.pad_texture(wObj, color)
        wScene.append(wObj)

    left_hand_verts, left_hand_faces, _ = sided_model.hand_para2verts_faces_joints(
        torch.FloatTensor(seq_data["left_hand"]["theta"]).to(device),
        torch.FloatTensor(seq_data["left_hand"]["shape"]).to(device),
        side="left",
    )
    left_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(device)
    color = torch.zeros_like(left_meshes.verts_padded())
    color[..., 1] = 1
    left_meshes.textures = mesh_utils.pad_texture(left_meshes, color)
    right_hand_verts, right_hand_faces, _ = sided_model.hand_para2verts_faces_joints(
        torch.FloatTensor(seq_data["right_hand"]["theta"]).to(device),
        torch.FloatTensor(seq_data["right_hand"]["shape"]).to(device),
        side="right",
    )
    right_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(device)
    color = torch.zeros_like(right_meshes.verts_padded())
    color[..., 2] = 1
    right_meshes.textures = mesh_utils.pad_texture(right_meshes, color)

    wScene += [left_meshes, right_meshes]
    wScene = mesh_utils.join_scene(wScene)
    wTc = torch.FloatTensor(seq_data["wTc"]).to(device)
    cTw = geom_utils.inverse_rt_v2(wTc)
    cScene = mesh_utils.apply_transform(wScene, cTw)

    intr = torch.FloatTensor(seq_data["intrinsic"]).to(device)
    down = 4
    H = W = 1408
    ndc_intr = mesh_utils.intr_from_screen_to_ndc(intr, W, H)
    fxfy, pxpy = mesh_utils.get_fxfy_pxpy(ndc_intr[None].repeat(T, 1, 1))
    cameras = PerspectiveCameras(fxfy, pxpy, device=device)

    # Use HardFlatShader for unlit/albedo rendering (no lighting effects)
    # Set up ambient-only lighting: ambient=1.0, diffuse=0.0, specular=0.0
    # This ensures textures are rendered at full brightness without any lighting variations
    ambient_lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),  # Full ambient brightness
        diffuse_color=((0.0, 0.0, 0.0),),  # No diffuse lighting
        specular_color=((0.0, 0.0, 0.0),),  # No specular highlights
    )
    unlit_materials = Materials(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),  # Full ambient reflectivity
        diffuse_color=((0.0, 0.0, 0.0),),  # No diffuse reflectivity
        specular_color=((0.0, 0.0, 0.0),),  # No specular reflectivity
        shininess=0,
    )
    
    flat_shader = HardFlatShader(
        device=device,
        lights=ambient_lights,
        materials=unlit_materials,
    )
    image = mesh_utils.render_mesh(
        cScene,
        cameras,
        out_size=(H // down, W // down),
        shader=flat_shader,
    )
    rgb = image["image"]  # (N, 3, H, W)
    fg_mask = image["mask"]
    left_hand_mask = (image["image"][:, 1:2] > 0.5).float()
    right_hand_mask = (image["image"][:, 2:3] > 0.5).float()
    # not left hand and not right hand
    obj_mask = fg_mask * (1 - left_hand_mask) * (1 - right_hand_mask)
    image_utils.save_gif(rgb.unsqueeze(1), osp.join(vis_dir, "rgb"), ext=".mp4", fps=30)

    # Discretize RGB image to object UIDs
    obj_mask = discretize_rgb_to_obj_uid(
        rgb, index, index2objid, obj_mask
    )  # (N, #obj, H, W)
    num_obj = obj_mask.shape[1]
    image_utils.save_gif(
        obj_mask.unsqueeze(2), osp.join(vis_dir, "obj_mask"), ext=".mp4", col=num_obj, max_size=324*8, fps=30
    )


def run_foundation_pose(seq):
    # save
    return


if __name__ == "__main__":
    hand_wrapper = HandWrapper(mano_model_folder).to(device)
    object_library = Pt3dVisualizer.setup_template(
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/", lib="hotclip"
    )
    get_mask(all_dat[seq_name], object_library, hand_wrapper)
