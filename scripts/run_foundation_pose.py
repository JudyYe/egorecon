from scipy.spatial.transform import Slerp, Rotation

# run foundation pose on HOT3D-CLIP
import os.path as osp
import json
from tqdm import tqdm
import os
from glob import glob
import numpy as np
import torch
from jutils import geom_utils, image_utils, mesh_utils
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shader import HardFlatShader
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

from egorecon.manip.data.utils import load_pickle
from egorecon.utils.motion_repr import HandWrapper
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer

mano_model_folder = "assets/mano"
object_mesh_dir = "data/HOT3D-CLIP/object_models_eval/"

seq_name = "002240"
data_dir = "data/HOT3D-CLIP/"
all_data = load_pickle("data/HOT3D-CLIP/preprocess/dataset_contact.pkl")
device = torch.device("cuda:0")


# vis_dir = "outputs/debug_fp_mask"
vis_dir = "outputs/debug_fp_pose_unidepth"


def batch_prorcess(func, split, skip=True, save_dir=""):
    split_file = osp.join("data/HOT3D-CLIP/sets", "split.json")
    with open(split_file, "r") as f:
        split_dict = json.load(f)
    seq_list = split_dict[split]
    for seq in tqdm(seq_list):
        done_file = osp.join(save_dir, "done", f"{seq}")
        lock_file = osp.join(save_dir, "lock", f"{seq}")
        save_file = osp.join(save_dir, f"{seq}.npz")

        if skip and osp.exists(done_file):
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            if skip:
                continue
        func(seq, save_file=save_file)
        os.makedirs(done_file)
        os.rmdir(lock_file)


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


def get_mask(
    seq, all_data, object_library, sided_model: HandWrapper, vis=False, save_file=None
):
    # mask:
    # {'uid': [T, H, W]}
    # create scene
    seq_data = all_data[seq]
    wScene = []

    index = (np.linspace(0.1, 0.9, len(seq_data["objects"])) * 255).astype(np.uint8)
    objid2index = {}
    for i, obj_id in enumerate(seq_data["objects"]):
        objid2index[obj_id] = index[i]
    index2objid = {index: obj_id for obj_id, index in objid2index.items()}

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
    left_hand_mask = ((image["image"][:, 1:2] > 0.5) & (fg_mask > 0.5)).float()
    right_hand_mask = ((image["image"][:, 2:3] > 0.5) & (fg_mask > 0.5)).float()
    # not left hand and not right hand
    obj_mask = fg_mask * (1 - left_hand_mask) * (1 - right_hand_mask)
    if vis:
        image_utils.save_gif(
            rgb.unsqueeze(1), osp.join(vis_dir, "rgb"), ext=".mp4", fps=30
        )

    # Discretize RGB image to object UIDs
    obj_mask = discretize_rgb_to_obj_uid(
        rgb, index, index2objid, obj_mask
    )  # (N, #obj, H, W)
    num_obj = obj_mask.shape[1]
    hand_obj_mask = torch.cat(
        [left_hand_mask, right_hand_mask, obj_mask], dim=1
    )  # (N, 2 + #obj, H, W)
    if vis:
        image_utils.save_gif(
            hand_obj_mask.unsqueeze(2),
            osp.join(vis_dir, "hand_obj_mask"),
            ext=".mp4",
            col=num_obj + 2,
            max_size=324 * 8,
            fps=30,
        )

    hand_obj_mask_np = (
        hand_obj_mask.transpose(0, 1).cpu().numpy() > 0.5
    )  # (2 + #obj, N, H, W)
    index = np.array(["left_hand", "right_hand", *seq_data["objects"].keys()])

    if save_file is None:
        save_file = osp.join(vis_dir, f"{seq}.npz")
    np.savez_compressed(save_file, hand_obj_mask=hand_obj_mask_np, index=index)


def run_foundation_pose(seq, all_data, object_library, save_file=None, vis=False):
    seq_key = seq
    selected_objects = None
    if seq not in all_data:
        if "_" in seq:
            parts = seq.split("_", 1)
            base_seq = parts[0]
            if base_seq in all_data:
                seq_key = base_seq
                selected_objects = [parts[1]]
            else:
                raise KeyError(f"Sequence {seq} not found in all_data and base sequence {base_seq} missing as well")
        else:
            raise KeyError(f"Sequence {seq} not found in all_data")

    seq_data = all_data[seq_key]
    object_ids = selected_objects or list(seq_data["objects"].keys())
    if len(object_ids) == 0:
        print(f"No objects available for sequence {seq_key}. Skipping.")
        return

    image_dir = osp.join(data_dir, "extract_images-rot90", f"clip-{seq_key}")
    image_list = sorted(glob(osp.join(image_dir, "*.jpg")))
    print(f"query {image_dir}, found {len(image_list)} images")
    T = len(image_list)
    assert T == len(seq_data["wTc"]), f"T mismatch: {T} != {len(seq_data['wTc'])}"

    mask_file = osp.join(data_dir, "gt_mask", f"{seq_key}.npz")
    mask_data = np.load(mask_file)
    hand_obj_mask = mask_data["hand_obj_mask"]
    index = mask_data["index"].tolist()

    wTc = torch.FloatTensor(seq_data["wTc"]).to(device)
    intr = torch.FloatTensor(seq_data["intrinsic"]).to(device)

    if vis:
        viz = Pt3dVisualizer(
            exp_name="log",
            save_dir=vis_dir,
            mano_models_dir=mano_model_folder,
            object_mesh_dir=object_mesh_dir,
        )
        sided_model = HandWrapper(mano_model_folder).to(device)
        left_hand_verts, left_hand_faces, _ = sided_model.hand_para2verts_faces_joints(
            torch.FloatTensor(seq_data["left_hand"]["theta"]).to(device),
            torch.FloatTensor(seq_data["left_hand"]["shape"]).to(device),
            side="left",
        )
        left_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(device)
        right_hand_verts, right_hand_faces, _ = (
            sided_model.hand_para2verts_faces_joints(
                torch.FloatTensor(seq_data["right_hand"]["theta"]).to(device),
                torch.FloatTensor(seq_data["right_hand"]["shape"]).to(device),
                side="right",
            )
        )
        right_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(device)
        hand_meshes = [left_meshes, right_meshes]

    wTo_list = []
    valid_list = []
    score_list = []
    processed_objects = []

    for obj_id in object_ids:
        if obj_id not in seq_data["objects"]:
            print(f"Object {obj_id} not present in sequence data for {seq_key}, skipping")
            continue
        if obj_id not in object_library:
            print(f"Object {obj_id} missing from object library, skipping")
            continue
        if obj_id not in index:
            print(f"Object {obj_id} not present in mask index for {seq_key}, skipping")
            continue

        mask_idx = index.index(obj_id)
        obj_mask = hand_obj_mask[mask_idx]
        intr = torch.FloatTensor(seq_data["intrinsic"]).to(device)
        intr[..., :2, :] /= down

        mesh_file = object_library[obj_id]
        textured_file = mesh_file.replace("/object_models_eval/", "/object_models/")
        intr_np = intr.cpu().numpy() if isinstance(intr, torch.Tensor) else intr
        if intr_np.shape == (3, 3):
            fx, fy = intr_np[0, 0], intr_np[1, 1]
            cx, cy = intr_np[0, 2], intr_np[1, 2]
            intr_np = np.array([fx, fy, cx, cy])

        cTo, valid, score, tracked = track_obj(textured_file, obj_mask, image_list, intr_np)
        wTo = wTc @ cTo
        # "001890_000009"
        # import pdb; pdb.set_trace()
        wTo = infill_obj(wTo, valid)

        wTo_list.append(wTo)
        valid_list.append(valid)
        score_list.append(score)
        processed_objects.append(obj_id)

        if vis:
            print('wTo', wTo.shape)
            oObj = mesh_utils.load_mesh(mesh_file).to(device)
            viz.log_hoi_step_from_cam_side(
                *hand_meshes,
                geom_utils.matrix_to_se3_v2(wTo),
                oObj,
                contact=None,
                pref=f"{seq_key}_{obj_id}_pred",
                intr_pix=intr,
                wTc=geom_utils.matrix_to_se3_v2(wTc),
                device=device,
                debug_info=[[f"frame {t}: {score[t]:.4f} {valid[t]:.1f} {tracked[t]:.1f}"] for t in range(len(wTo))],
            )
            wTo_gt = torch.FloatTensor(seq_data["objects"][obj_id]["wTo"]).to(device)
            viz.log_hoi_step_from_cam_side(
                *hand_meshes,
                geom_utils.matrix_to_se3_v2(wTo_gt),
                oObj,
                contact=None,
                pref=f"{seq_key}_{obj_id}_gt",
                intr_pix=intr,
                wTc=geom_utils.matrix_to_se3_v2(wTc),
                device=device,
            )

    if not processed_objects:
        print(f"No objects processed for sequence {seq}. Nothing saved.")
        return

    wTo_array = torch.stack(wTo_list, 0).cpu().numpy()
    valid_array = torch.stack(valid_list, 0).cpu().numpy()
    score_array = torch.stack(score_list, 0).cpu().numpy()
    print("wTo_list", wTo_array.shape, valid_array.shape, score_array.shape)

    if save_file is None:
        save_file = osp.join(vis_dir, f"{seq}.npz")
    os.makedirs(osp.dirname(save_file), exist_ok=True)
    index_saved = np.array(["left_hand", "right_hand", *processed_objects])
    np.savez_compressed(
        save_file,
        wTo=wTo_array,
        valid=valid_array,
        index=index_saved,
        score=score_array,
    )

    print(f"Saved foundation pose to {save_file}")
    return


def vis_fp(index, all_data, object_library):

    seq, obj_id = index.split("_")
    seq_data = all_data[seq]

    fname = osp.join(data_dir, "foundation_pose", f"{seq}.npz")
    fp_data = np.load(fname)
    
    wTo_list = torch.FloatTensor(fp_data["wTo"]).to(device)
    hand_obj_list = fp_data["index"].tolist()
    score_list = fp_data["score"]
    valid_list = fp_data["valid"]
    print('hand_obj_list', hand_obj_list, score_list.shape)

    obj_idx = hand_obj_list.index(obj_id) - 2
    wTo = wTo_list[obj_idx]

    oObj = mesh_utils.load_mesh(object_library[obj_id]).to(device)
    intr = torch.FloatTensor(seq_data["intrinsic"]).to(device)
    wTc = torch.FloatTensor(seq_data["wTc"]).to(device)

    viz = Pt3dVisualizer(
        exp_name="log",
        save_dir=vis_dir,
        mano_models_dir=mano_model_folder,
        object_mesh_dir=object_mesh_dir,
    )
    sided_model = HandWrapper(mano_model_folder).to(device)
    left_hand_verts, left_hand_faces, _ = sided_model.hand_para2verts_faces_joints(
        torch.FloatTensor(seq_data["left_hand"]["theta"]).to(device),
        torch.FloatTensor(seq_data["left_hand"]["shape"]).to(device),
        side="left",
    )
    left_meshes = Meshes(verts=left_hand_verts, faces=left_hand_faces).to(device)
    right_hand_verts, right_hand_faces, _ = sided_model.hand_para2verts_faces_joints(
        torch.FloatTensor(seq_data["right_hand"]["theta"]).to(device),
        torch.FloatTensor(seq_data["right_hand"]["shape"]).to(device),
        side="right",
    )
    right_meshes = Meshes(verts=right_hand_verts, faces=right_hand_faces).to(device)
    hand_meshes = [left_meshes, right_meshes]

    text_list = []
    for t in range(wTo.shape[0]):
        text_list.append([f"frame {t}: {score_list[obj_idx][t]:.4f} {valid_list[obj_idx][t]:.1f}"])
    viz.log_hoi_step_from_cam_side(
        *hand_meshes,
        geom_utils.matrix_to_se3_v2(wTo),
        oObj,
        contact=None,
        pref=f"{seq}_{obj_id}_pred",
        intr_pix=intr,
        wTc=geom_utils.matrix_to_se3_v2(wTc),
        device=device,
        debug_info=text_list
    )


def infill_obj(cTo, valid):
    """_summary_
    cTo: (T, 4, 4)
    valid: (T,)
    return: (T, 4, 4)
    """
    device = cTo.device
    valid = valid.cpu().numpy().astype(np.bool_)
    T = cTo.shape[0]
    rot, tsl, _ = geom_utils.homo_to_rt(cTo, ignore_scale=True)
    tsl = tsl.cpu().numpy()

    quat = matrix_to_quaternion(rot)  # wxyz  # T, 4
    quat = quat.cpu().numpy()

    quat = slerp_interpolation_quat(quat.reshape(1, T, 1, 4), valid.reshape(1, T))
    quat = torch.FloatTensor(quat).reshape(T, 4)  # wxyz
    rot = quaternion_to_matrix(quat)

    tsl = linear_interpolation_nd(tsl.reshape(1, T, 3), valid.reshape(1, T))
    tsl = torch.FloatTensor(tsl).reshape(T, 3)

    cTo = geom_utils.rt_to_homo(rot, tsl).to(device)
    return cTo


def slerp_interpolation_aa(pos, valid):
    B, T, N, _ = pos.shape  # B: 批次大小, T: 时间步长, N: 关节数, 4: 四元数维度
    pos_interp = pos.copy()  # 创建副本以存储插值结果

    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]
            valid_b_n = valid[b, :]

            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0:
                continue

            if len(valid_idxs) > 1:
                valid_times = valid_idxs  # 有效时间步
                valid_rots = Rotation.from_rotvec(quat_b_n[valid_idxs])  # 有效四元数

                slerp = Slerp(valid_times, valid_rots)

                for idx in invalid_idxs:
                    if idx < valid_idxs[0]:  # 时间步小于第一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[
                            valid_idxs[0]
                        ]  # 复制第一个有效四元数
                    elif idx > valid_idxs[-1]:  # 时间步大于最后一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[
                            valid_idxs[-1]
                        ]  # 复制最后一个有效四元数
                    else:
                        interp_rot = slerp([idx])
                        pos_interp[b, idx, n, :] = interp_rot.as_rotvec()[0]

            if len(valid_idxs) == 1:
                # just repeat the only valid value for the whole sequence
                pos_interp[b, :, n, :] = quat_b_n[valid_idxs[0]]

    # print("#######")
    # if N > 1:
    #     print(pos[1,0,11])
    #     print(pos_interp[1,0,11])

    return pos_interp


def slerp_interpolation_quat(pos, valid):
    """_summary_
    :param pos: (B, T, N, D) wxyz
    :param valid: (B, T)
    :return: (B, T, N, D)
    """
    # wxyz to xyzw
    pos = pos[:, :, :, [1, 2, 3, 0]]

    B, T, N, _ = pos.shape  # B: 批次大小, T: 时间步长, N: 关节数, 4: 四元数维度
    pos_interp = pos.copy()  # 创建副本以存储插值结果

    for b in range(B):
        for n in range(N):
            quat_b_n = pos[b, :, n, :]
            valid_b_n = valid[b, :]
            print(valid_b_n.dtype, valid_b_n.shape)

            invalid_idxs = np.where(~valid_b_n)[0]
            valid_idxs = np.where(valid_b_n)[0]

            if len(invalid_idxs) == 0:
                continue

            if len(valid_idxs) > 1:
                valid_times = valid_idxs  # 有效时间步
                valid_rots = Rotation.from_quat(quat_b_n[valid_idxs])  # 有效四元数

                slerp = Slerp(valid_times, valid_rots)

                for idx in invalid_idxs:
                    if idx < valid_idxs[0]:  # 时间步小于第一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[
                            valid_idxs[0]
                        ]  # 复制第一个有效四元数
                    elif idx > valid_idxs[-1]:  # 时间步大于最后一个有效时间步，进行外推
                        pos_interp[b, idx, n, :] = quat_b_n[
                            valid_idxs[-1]
                        ]  # 复制最后一个有效四元数
                    else:
                        interp_rot = slerp([idx])
                        pos_interp[b, idx, n, :] = interp_rot.as_quat()[0]

    # xyzw to wxyz
    pos_interp = pos_interp[:, :, :, [3, 0, 1, 2]]
    return pos_interp


def linear_interpolation_nd(pos, valid, start_end=True):
    """_summary_"""
    B, T = pos.shape[:2]  # 取出批次大小B和时间步长T
    feature_dim = pos.shape[2]  # ** 代表的任意维度
    pos_interp = pos.copy()  # 创建一个副本，用来保存插值结果

    for b in range(B):
        for idx in range(feature_dim):  # 针对任意维度
            pos_b_idx = pos[b, :, idx]  # 取出第b批次对应的**维度下的一个时间序列
            valid_b = valid[b, :]  # 当前批次的有效标志

            # 找到无效的索引（False）
            invalid_idxs = np.where(~valid_b)[0]
            valid_idxs = np.where(valid_b)[0]

            if len(invalid_idxs) == 0:
                continue

            # 对无效部分进行线性插值
            if len(valid_idxs) > 1:  # 确保有足够的有效点用于插值
                pos_b_idx[invalid_idxs] = np.interp(
                    invalid_idxs, valid_idxs, pos_b_idx[valid_idxs]
                )
                pos_interp[b, :, idx] = pos_b_idx  # 保存插值结果

            # NOTE: judy's code
            if start_end and len(valid_idxs) == 1:
                # just repeat the only valid value for the whole sequence
                # print('before', pos_interp[b, :, idx])
                pos_interp[b, :, idx] = pos_b_idx[valid_idxs[0]]
                # print('after', pos_interp[b, :, idx])
            # import ipdb; ipdb.set_trace()

    return pos_interp


def track_obj(mesh_file, obj_mask, image_list, intr):
    """
    Track object pose using FoundationPose with automatic reinitialization.

    Args:
        fp_wrapper: PoseWrapper instance (optional, will create new if None)
        mesh_file: Path to object mesh file (.glb or .obj)
        obj_id: Object ID (for logging)
        obj_mask: (T, H, W) boolean mask array
        image_list: List of image paths
        intr: Camera intrinsics [fx, fy, cx, cy] or 3x3 matrix

    Returns:
        cTo: (T, 4, 4) array of poses in camera frame
        valid: (T,) boolean array indicating valid poses
    """
    from scripts.fp_utils import PoseWrapper

    pose_wrapper = PoseWrapper(mesh_file=mesh_file, intrinsic=intr, mask_list=obj_mask)

    # Track through video
    # cTo, valid, score, tracked = pose_wrapper.track_video(image_list, intr)
    cTo, valid, score, tracked = pose_wrapper.track_video_from_best_frame(image_list, intr)
    cTo = torch.FloatTensor(cTo).to(device)
    valid = torch.FloatTensor(valid).to(device)
    score = torch.FloatTensor(score).to(device)
    print("cTo", cTo.shape, valid.shape, score.shape)

    return cTo, valid, score, tracked


down = 4

if __name__ == "__main__":
    hand_wrapper = HandWrapper(mano_model_folder).to(device)
    object_library = Pt3dVisualizer.setup_template(
        object_mesh_dir="data/HOT3D-CLIP/object_models_eval/", lib="hotclip"
    )

    # func = partial(get_mask, object_library=object_library, sided_model=hand_wrapper, all_data=all_data)

    # batch_prorcess(func,  split="test", save_dir=osp.join(data_dir, "gt_mask"))
    # get_mask("001870", all_data, object_library, hand_wrapper, save_file=osp.join(data_dir, "gt_mask", "001870.npz"))

    object_library = Pt3dVisualizer.setup_template(
        object_mesh_dir=object_mesh_dir,
        lib="hotclip",
        load_mesh=False,
    )
    # seq_name = "001992"
    # seq_name = "001890_000009"
    # run_foundation_pose(
    #     seq_name,
    #     all_data,
    #     object_library,
    #     save_file=osp.join(vis_dir, f"{seq_name}.npz"),
    #     vis=True,
    # )
    from functools import partial
    from scripts.fp_utils import depth_model

    func = partial(
        run_foundation_pose,
        object_library=object_library,
        all_data=all_data,
        vis=True
    )
    # batch_prorcess(func, split="test50obj", save_dir=osp.join(data_dir, f"foundation_pose_{depth_model}"))
    batch_prorcess(func, split="teaser", save_dir=osp.join(data_dir, f"foundation_pose_{depth_model}"))

    # seq = "002862"
    # obj_list = ['000002', '000006', '000011', '000017', '000003', '000004']
    # # index = "002862_000006"
    # for obj in obj_list:
    #     index = f"{seq}_{obj}"
    #     vis_fp(index, all_data, object_library)