import open3d as o3d
import argparse
import json
import os
import os.path as osp
import pickle
import tarfile
from collections import defaultdict
from glob import glob
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rerun as rr
import torch
from hand_tracking_toolkit.dataset import HandShapeCollection, warp_image
from jutils import hand_utils
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import preprocess.hot3dclip_util as clip_util  # pyre-ignore
from hot3d.hot3d.data_loaders.loader_hand_poses import Handedness
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer
from jutils import mesh_utils
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

# data format:
# seq:
# "objects": ['uid1', 'uid2', ...],
# "left_hand_theta": [T, 3+3+15], 3aa+3tsl+pcax15
# "left_hand_shape": [T, 10],
# "right_hand_theta": [T, 3+3+15],
# "right_hand_shape": [T, 10],
# "wTc": [T, 4, 4],
# intrinsic: [3, 3],
# "obj_{i}_cTo_shelf": [T, 4, 4],
# "obj_{i}_shelf_valid": [T, ],
# "obj_{i}_wTo": [T, 4, 4],
# "obj_{i}_wTo_valid": [T, ],

def rotate_90(clip_path, **kwargs):
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    cur_image_dir = os.path.join(args.preprocess_dir.split('-rot90')[0], clip_name)
    dst_image_dir = os.path.join(args.preprocess_dir, clip_name)
    query = osp.join(dst_image_dir, '*.jpg')
    image_paths = glob(query)
    if len(image_paths) == len(glob(os.path.join(cur_image_dir, '*.jpg'))):
        print('already rotated')
        return
    print(query, len(image_paths), len(glob(os.path.join(cur_image_dir, '*.jpg'))))

    os.makedirs(dst_image_dir, exist_ok=True)
    
    for image_path in glob(os.path.join(cur_image_dir, '*.jpg')):
        image = cv2.imread(image_path)
        image = np.rot90(image, k=3)
        cv2.imwrite(os.path.join(dst_image_dir, os.path.basename(image_path)), image)
    return dst_image_dir


def extract_image_one_clip(clip_path, **kwargs):
    """
    Extract images from a clip and save them as %04d.jpg in extract_images/clip_name/
    
    Args:
        clip_path: Path to the .tar file containing the clip data
    """
    tar = tarfile.open(clip_path, mode="r")
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    
    # Create output directory for this clip
    extract_dir = os.path.join(args.preprocess_dir, clip_name)
    os.makedirs(extract_dir, exist_ok=True)
    
    T = clip_util.get_number_of_frames(tar)
    
    for frame_id in tqdm(range(T), desc=f"Extracting images from {clip_name}"):
        frame_key = f"{frame_id:06d}"
        
        # Load camera parameters
        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))
        
        # Use the first image stream
        stream_id = image_streams[0]
        stream_key = str(stream_id)
        
        # Load the image
        image: np.ndarray = clip_util.load_image(tar, frame_key, stream_key)
        
        # Make sure the image has 3 channels
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Get camera model for potential warping
        camera_model = cameras[stream_id]
        camera_model_orig = camera_model
        camera_model = clip_util.convert_to_pinhole_camera(camera_model)
        
        # Warp image if needed
        image = warp_image(
            src_camera=camera_model_orig,
            dst_camera=camera_model,
            src_image=image,
        )
        down_sample = 4
        H, W = image.shape[:2]
        image = cv2.resize(image, (W // down_sample, H // down_sample))
        
        # Save image with 4-digit zero-padded frame number
        output_path = os.path.join(extract_dir, f"{frame_id:04d}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    tar.close()
    print(f"Extracted {T} images to {extract_dir}")
    return extract_dir

    
device = 'cuda:0'
def get_one_clip(clip_path, vis=False):
    tar = tarfile.open(clip_path, mode="r")
    clip_name = os.path.basename(clip_path).split(".tar")[0]
    clip_output_path = os.path.join(vis_dir, clip_name)
    os.makedirs(clip_output_path, exist_ok=True)
    hand_shape: Optional[HandShapeCollection] = clip_util.load_hand_shape(tar)

    anno = defaultdict(list)
    #
    # anno = {
    #     "objects": [],
    #     "left_hand_theta": [],
    #     "left_hand_shape": [],
    #     "right_hand_theta": [],
    #     "right_hand_shape": [],
    #     "wTc": [],
    #     "intrinsic": [],
    #     "uid{i}_wTo": [],
    #     "uid{i}_wTo_valid": [T, ],
    # }
    T = clip_util.get_number_of_frames(tar)
    anno["wTc"] = np.tile(np.eye(4)[None], (T, 1, 1))
    anno["left_hand_shape"] = [hand_shape.mano_beta.cpu().numpy()] * T
    anno["right_hand_shape"] = [hand_shape.mano_beta.cpu().numpy()] * T
    anno["left_hand_theta"] = np.zeros((T, 3 + 3 + 15))
    anno["right_hand_theta"] = np.zeros((T, 3 + 3 + 15))
    image_list = []

    for frame_id in tqdm(range(clip_util.get_number_of_frames(tar))):
        frame_key = f"{frame_id:06d}"

        # Load camera parameters.
        cameras, _ = clip_util.load_cameras(tar, frame_key)
        image_streams = sorted(cameras.keys(), key=lambda x: int(x.split("-")[0]))

        # Load hand and object annotations.
        hands: Optional[Dict[str, Any]] = clip_util.load_hand_annotations(
            tar, frame_key
        )

        for hand_side in ["left", "right"]:
            if hand_side not in hands:
                print("Why no hand????", clip_name, frame_id, hand_side)
                assert False
            theta = np.array(hands[hand_side]["mano_pose"]["thetas"])  # 15
            xform = np.array(hands[hand_side]["mano_pose"]["wrist_xform"])  # 6
            global_orient = xform[:3]
            transl = xform[3:]
            thetas = np.concatenate([global_orient, transl, theta], axis=-1)
            anno[f"{hand_side}_hand_theta"][frame_id] = thetas

        objects: Optional[Dict[str, Any]] = clip_util.load_object_annotations(
            tar, frame_key
        )

        stream_id = image_streams[0]

        # Camera parameters of the current image.
        camera_model = cameras[stream_id]
        camera_model_orig = camera_model
        camera_model = clip_util.convert_to_pinhole_camera(camera_model)
        if vis:
            stream_key = str(stream_id)

            # Load the image.
            image: np.ndarray = clip_util.load_image(tar, frame_key, stream_key)

            # Make sure the image has 3 channels.
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)

            image = warp_image(
                src_camera=camera_model_orig,
                dst_camera=camera_model,
                src_image=image,
            )
            image_list.append(image)

        intrinsic = camera_model.uv_to_window_matrix()
        wTc = camera_model.T_world_from_eye
        rot = np.array(
            [
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        anno["wTc"][frame_id] = wTc @ rot

        if objects is None:
            print("Why????", clip_name, frame_id)
            assert False

        for instance_list in objects.values():
            for instance in instance_list:
                bop_id = int(instance["object_bop_id"])
                bop_id = f"{bop_id:06d}"
                wTo = clip_util.se3_from_dict(instance["T_world_from_object"])
                if f"obj_{bop_id}_wTo" not in anno:
                    anno[f"obj_{bop_id}_wTo"] = np.tile(np.eye(4)[None], (T, 1, 1))
                    anno[f"obj_{bop_id}_wTo_valid"] = np.zeros(T).astype(bool)

                anno[f"obj_{bop_id}_wTo"][frame_id] = wTo
                anno[f"obj_{bop_id}_wTo_valid"][frame_id] = True
                anno["objects"].append(bop_id)

    # get all bop_id
    bop_id_list = list(set(anno["objects"]))
    anno["objects"] = bop_id_list
    anno["intrinsic"] = intrinsic

    for key, value in anno.items():
        anno[key] = np.array(value)

    if vis:
        vis_meta(anno, image_list, clip_output_path)

    out_file = osp.join(args.preprocess_dir, clip_name + ".npz")
    np.savez_compressed(out_file, **anno)
    return anno, image_list


def vis_meta(meta, image_list, vis_dir, num=-1):
    if isinstance(meta, str):
        meta = dict(np.load(meta, allow_pickle=True))

    os.makedirs(vis_dir, exist_ok=True)
    intrinsic = meta["intrinsic"]

    T_w_c = meta["wTc"]

    if image_list is not None:
        H, W = image_list[0].shape[:2]
    else:
        H, W = intrinsic[1, 1] * 2, intrinsic[0, 0] * 2

    rr.init("vis_meta")
    rr.save(osp.join(vis_dir, f"vis_meta.rrd"))
    print("Saved rrd to", osp.join(vis_dir, f"vis_meta.rrd"))
    print(meta["objects"])

    sided_wrapper = {}
    for side in ["left", "right"]:
        sided_wrapper[side] = hand_utils.ManopthWrapper(
            args.mano_model_dir,
            side=side,
        )

    if num < 0:
        num = len(T_w_c)

    for t in range(num):
        rr.set_time_sequence("frame", t)
        # log image
        if image_list is not None:
            image = image_list[t]
            # resacle to maxium 256
            image = cv2.resize(image, (256, 256))
            image = np.rot90(image, k=3)
            rr.log("world/image", rr.Image(image))
        pose = T_w_c[t]
        focal_length = intrinsic[0, 0]
        if t == 0:
            rr.log(
                "world/camera",
                rr.Pinhole(
                    width=W,
                    height=H,
                    focal_length=float(focal_length),
                ),
            )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=pose[:3, 3],
                rotation=rr.Quaternion(
                    xyzw=Rotation.from_matrix(pose[:3, :3]).as_quat()
                ),
            ),
        )
        for uid in meta["objects"]:
            new_k = f"obj_{uid}_wTo"
            
            # new_k = k.replace("uid", "obj_")
            # new_k = new_k.replace("gt_valid", "wTo_valid")
            # new_meta[new_k] = meta[k]

            k = new_k.replace("obj_", "uid")
            k = k.replace("wTo_valid", "gt_valid")

            if new_k not in meta:
                new_k = k
                
            # T_w_o = meta[f"obj_{uid}_wTo"][t]
            T_w_o = meta[new_k][t]
            mesh_file = osp.join(args.object_models_dir, f"obj_{uid}.glb")

            # Check if object has contact with any hand
            contact_key = f"obj_{uid}_contact_lr"
            has_any_contact = False
            if contact_key in meta:
                contact_lr = meta[contact_key][t]  # Shape: (2,)
                has_any_contact = np.any(contact_lr)
            if t > 0 and contact_key in meta:
                prev_any_contact = np.any(meta[contact_key][t-1])
            

            if t == 0 or contact_key in meta and (has_any_contact != prev_any_contact):
            #     rr.log(
            #         f"world/object_pose/{uid}",
            #         rr.Asset3D(
            #             path=mesh_file,
            #         ),
            #     )
                mesh = o3d.io.read_triangle_mesh(mesh_file)
                verts = np.array(mesh.vertices)
                faces = np.array(mesh.triangles)

                if has_any_contact:
                    color = np.array([[1.0, 0.0, 0.0]])
                else:
                    color = np.array([[0.0, 1.0, 0.0]])
                color = np.tile(color, (verts.shape[0], 1))
                # remove previous object mesh

                # add the new colored mesh
                rr.log(
                    f"world/object_pose/{uid}",
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces,
                        vertex_colors=(color * 255).astype(np.uint8),
                    ),
                )

                
            rr.log(
                f"world/object_pose/{uid}",
                rr.Transform3D(
                    translation=T_w_o[:3, 3],
                    rotation=rr.Quaternion(
                        xyzw=Rotation.from_matrix(T_w_o[:3, :3]).as_quat()
                    ),
                ),
            )
            # make color 
            # maybe just log contact_lr as scalar to monitor its value

        # hand pose
        for h, handedness in enumerate(["left", "right"]):
            hand_pose = meta[f"{handedness}_hand_theta"][t]
            shape = meta[f"{handedness}_hand_shape"][t]
            if handedness == "left":
                side = Handedness.Left
            else:
                side = Handedness.Right
            # faces = hand_data_provider.get_hand_mesh_faces_and_normals(side)[0]

            rot, tsl, hA = hand_pose[..., :3], hand_pose[..., 3:6], hand_pose[..., 6:]

            rot = torch.FloatTensor(rot)[None]
            tsl = torch.FloatTensor(tsl)[None]
            hA = torch.FloatTensor(hA)[None]
            shape = torch.FloatTensor(shape)[None]
            hA = sided_wrapper[handedness].pca_to_pose(hA)

            mesh, _ = sided_wrapper[handedness](
                None, hA, axisang=rot, trans=tsl, th_betas=shape
            )
            faces = sided_wrapper[handedness].hand_faces.cpu().numpy()[0]
            vertices = mesh.verts_packed().cpu().numpy()
            any_contact = False
            for uid in meta["objects"]:
                if f"obj_{uid}_contact_lr" in meta and meta[f"obj_{uid}_contact_lr"][t][h]:
                    any_contact = True
                    break
            if any_contact:
                color = np.array([[1.0, 0.0, 0.0]])  # Red for contact
            else:
                color = np.array([[0.0, 1.0, 0.0]])  # Green for no contact
            color = np.tile(color, (vertices.shape[0], 1))
            rr.log(
                f"world/hand/{handedness}",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=faces,
                    vertex_colors=(color * 255).astype(np.uint8),
                ),
            )
    return


def batch_preprocess(func):
    # clips = sorted([p for p in os.listdir(args.clips_dir, 'train_aria') if p.endswith(".tar")])
    clips_aria = sorted(glob(os.path.join(args.clips_dir, "train_aria", "*.tar")))
    clips_quest = sorted(glob(os.path.join(args.clips_dir, "train_quest3", "*.tar")))
    clips = clips_aria + clips_quest
    for clip in tqdm(clips):
        lock_file = osp.join(args.preprocess_dir, "lock", os.path.basename(clip))
        done_file = osp.join(args.preprocess_dir, "done", os.path.basename(clip))

        if osp.exists(done_file):
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            continue

        # get_one_clip(clip, vis=False)
        func(clip, vis=False)

        os.makedirs(done_file)
        os.rmdir(lock_file)



def find_missing_contact():
    data_file = osp.join(args.preprocess_dir, "dataset.pkl")
    with open(data_file, "rb") as f:
        data = pickle.load(f)

    data_dir = osp.join(args.preprocess_dir, "dataset_contact")
    for s, seq in enumerate(tqdm(data.keys())):
        obj_list = data[seq]["objects"]
        npz_file = osp.join(data_dir, f"{seq}.npz")
        flag = False


        try:
            new_data = np.load(npz_file, allow_pickle=True)
            new_data = dict(new_data)
            for obj in obj_list:
                key = f"obj_{obj}_contact_lr"
                if key not in new_data:
                    print(f"Missing contact for {seq}, {obj}")
                    flag = True
                    break
        except EOFError:
            print(f"EOFError {seq}")
            flag = True

        done_file = osp.join(data_dir, "done", f"{seq}.done")
        if flag:
            os.system('rm -rf ' + done_file)
        else:
            os.makedirs(done_file, exist_ok=True)
        #         continue
        # key = f"obj_{obj_list[0]}_contact_lr"
        # if key not in data[seq]:
        #     print(f"Missing contact for {seq}")
        #     continue



@torch.no_grad()
def patch_contact(contact_th=0.01):
    # per-frame contact label: yes or no, threshold is contact_th
    data_file = osp.join(args.preprocess_dir, "dataset.pkl")
    obj_library = Pt3dVisualizer.setup_template(object_mesh_dir=args.object_models_dir)

    sided_mano_model = {}
    for side in ["left", "right"]:
        sided_mano_model[side] = hand_utils.ManopthWrapper(
            args.mano_model_dir,
            side=side,
        ).to(device)

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    for s, seq in enumerate(tqdm(data.keys())):
        save_file = osp.join(args.preprocess_dir, "dataset_contact", f"{seq}.npz")
        lock_file = osp.join(args.preprocess_dir, "dataset_contact", "lock", f"{seq}.lock")
        done_file = osp.join(args.preprocess_dir, "dataset_contact", "done", f"{seq}.done")

        if osp.exists(done_file):
            print(f"Done {seq}")
            continue
        try:
            os.makedirs(lock_file)
        except FileExistsError:
            continue

        for obj in data[seq]["objects"]:
            wTo = torch.FloatTensor(data[seq][f"obj_{obj}_wTo"]).to(device) # (T, 4, 4)
            T = len(wTo)
            oObject = obj_library[obj].to(device)

            verts = oObject.verts_padded().repeat(T, 1, 1)
            faces = oObject.faces_padded().repeat(T, 1, 1)
            oObject = Meshes(verts=verts, faces=faces)
            print(wTo.shape, len(oObject))

            wObj = mesh_utils.apply_transform(oObject, wTo)

            hand_side = {}
            contact = []

            for side in ["left", "right"]:
                hand_pose = torch.FloatTensor(data[seq][f"{side}_hand_theta"]).to(device)
                hand_shape = torch.FloatTensor(data[seq][f"{side}_hand_shape"]).to(device)
                rot, tsl, hA = hand_pose[..., :3], hand_pose[..., 3:6], hand_pose[..., 6:]
                assert hA.shape[-1] == 15
                hA = sided_mano_model[side].pca_to_pose(hA)

                wHand, _ = sided_mano_model[side](None, hA, axisang=rot, trans=tsl, th_betas=hand_shape)
                hand_side[side] = wHand

                num_points = 10000
                num_points_hands = 1000

                wObj_points  = sample_points_from_meshes(wObj, num_points)
                wHand_points = sample_points_from_meshes(wHand, num_points_hands)

                dist = torch.cdist(wHand_points, wObj_points)  # (T, num_points_hands, num_points_obj)
                print(dist.shape)
                T = len(wObj)
                cdist = dist.reshape(T, -1).min(dim=-1)[0] < contact_th
                contact.append(cdist.cpu().numpy())  # 
            contact = np.stack(contact, axis=-1)  # (T, 2)
            key = f'obj_{obj}_contact_lr'
            data[seq][key] = contact

            np.savez_compressed(save_file, **data[seq])
            os.makedirs(done_file, exist_ok=True)
            os.system('rm -rf ' + lock_file)

            # data[seq][f"{side}_hand_contact"] = contact[side]
        # vis_meta(data[seq], None, vis_dir + f"/{s}")
        # if s > 0:
        #     break
            
    # new_data_file = osp.join(args.preprocess_dir, "dataset_contact.pkl")

    # with open(new_data_file, "wb") as f:
    #     pickle.dump(data, f)

    return


def merge_preprocess():
    preprocess_dir = args.preprocess_dir
    meta_list = glob(osp.join(preprocess_dir, "*.npz"))
    data = {}
    for meta_file in meta_list:
        seq = os.path.basename(meta_file).split(".")[0].split("-")[-1]
        # Use context manager to ensure file is properly closed
        with np.load(meta_file) as npz_file:
            meta = dict(npz_file)
        # change uid* to obj_*
        new_meta = {}
        for k in meta.keys():
            # if k.startswith("uid"):
            new_k = k.replace("uid", "obj_")
                # _gt_valid -> _wTo_valid
            new_k = new_k.replace("gt_valid", "wTo_valid")
            new_meta[new_k] = meta[k]
        
        data[seq] = new_meta

    with open(osp.join(preprocess_dir, "dataset.pkl"), "wb") as f:
        pickle.dump(data, f)
    return

def make_own_split():
    orig_split_file = osp.join(args.clips_dir, "clip_splits.json")
    with open(orig_split_file, "r") as f:
        orig_split = json.load(f)
    all_clips  = orig_split['train']['Aria'] + orig_split['train']['Quest3']
    clip_definition_file = osp.join(args.clips_dir, "clip_definitions.json")
    with open(clip_definition_file, "r") as f:
        clip_def = json.load(f)
    clipid2seqid = {int(k): v['sequence_id'] for k, v in clip_def.items()}

    from preprocess.hot3d import hawor_eval_seqs
    test_seqs = set(hawor_eval_seqs)


    split = {
        'train': [],
        'test': [],
    }
    for clip in all_clips:
        seqid = clipid2seqid[clip]
        if seqid in test_seqs:
            split['test'].append(f"{clip:06d}")
        else:
            split['train'].append(f"{clip:06d}")
    print('Total clips:', len(all_clips), '-->', len(split['train']), len(split['test']))
    set_dir = osp.join(args.clips_dir, "sets")
    os.makedirs(set_dir, exist_ok=True)
    with open(osp.join(set_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=4)
    return 



mano_model_folder = "hot3d/hot3d/mano_v1_2/models"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips_dir",
        type=str,
        help="Path to a folder with clips.",
        default="data/HOT3D-CLIP",
    )
    parser.add_argument(
        "--object_models_dir",
        type=str,
        help="Path to a folder with 3D object models.",
        default="data/HOT3D-CLIP/object_models_eval/",
    )
    parser.add_argument(
        "--mano_model_dir",
        type=str,
        help="Path to a folder with MANO model (MANO_RIGHT/LEFT.pkl files).",
        default="assets/mano/",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="data/HOT3D-CLIP/preprocess/",
    )
    args = parser.parse_args()
    vis_dir = "outputs/vis_hot3d_clips_preprocess/"

    
    # meta_file = "/move/u/yufeiy2/HaWoR/example/clip-002354-rot90/hawor_gtTrue.npz"
    # meta = dict(np.load(meta_file, allow_pickle=True))
    # vis_meta(meta, None, osp.join(vis_dir, 'hawor'))


    # meta_file = "data/HOT3D-CLIP/preprocess/clip-002354.npz"
    # meta = dict(np.load(meta_file, allow_pickle=True))
    # vis_meta(meta, None, osp.join(vis_dir, 'gt'))


    # batch_preprocess(extract_image_one_clip)
    batch_preprocess(rotate_90)

    # find_missing_contact()
    # patch_contact()
    # merge_preprocess()

    # make_own_split()



    # Example usage of extract_image_one_clip
    # Uncomment and modify the path to test:

    # clip_path = osp.join(args.clips_dir, "train_aria", "clip-002240.tar")
    # rotate_90(clip_path)
    # extract_image_one_clip(clip_path)


