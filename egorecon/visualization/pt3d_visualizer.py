import json
import os
import os.path as osp
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import pytorch3d
import torch
from jutils import geom_utils, image_utils, mesh_utils, plot_utils
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.transforms import euler_angles_to_matrix


class Pt3dVisualizer:
    def __init__(
        self,
        exp_name,
        save_dir,
        enable_visualization=True,
        mano_models_dir="data/mano_models",
        object_mesh_dir="data/object_meshes",
    ):
        self.exp_name = exp_name
        self.save_dir = Path(save_dir) / "log"
        self.enable_visualization = enable_visualization
        self.mano_models_dir = Path(mano_models_dir)
        self.object_mesh_dir = Path(object_mesh_dir)
        self.object_cache = self.setup_template(self.object_mesh_dir)

    @staticmethod
    def setup_template(object_mesh_dir, lib="hotclip", load_mesh=True):
        object_cache = {}
        if lib == "hot3d":
            glob_file = glob(osp.join(object_mesh_dir, "*.glb"))
            for mesh_file in glob_file:
                uid = osp.basename(mesh_file).split(".")[0]
                object_cache[uid] = mesh_utils.load_mesh(mesh_file)
        elif lib == "hotclip":
            # make it compatible with hot3d
            glob_file = glob(osp.join(object_mesh_dir, "*.glb"))

            model_info = json.load(open(osp.join(object_mesh_dir, "models_info.json")))
            obj2uid = {}
            for obj_id, obj_info in model_info.items():
                obj2uid[int(obj_id)] = obj_info["original_id"]
            for mesh_file in glob_file:
                obj_id = int(osp.basename(mesh_file).split(".")[0].split("_")[-1])
                if load_mesh:
                    mesh = mesh_utils.load_mesh(mesh_file)
                else:
                    mesh = mesh_file
                object_cache[f"{obj_id:06d}"] = mesh
                object_cache[obj2uid[obj_id]] = object_cache[f"{obj_id:06d}"]
        else:
            raise ValueError(f"Invalid library: {lib}")
        return object_cache

    @staticmethod
    def get_scene(
        left_hand_meshes, right_hand_meshes, wTo, oObj, contact, device, coord=True
    ):
        B, P, D = left_hand_meshes.verts_padded().shape
        left_textures = mesh_utils.get_color("blue").expand(B, P, 3).to(device)
        right_textures = mesh_utils.get_color("blue").expand(B, P, 3).to(device)
        _, V, D = oObj.verts_padded().shape
        obj_textures = mesh_utils.get_color("white").expand(B, V, 3).to(device)

        if contact is not None:
            contact_hand_textures = (
                mesh_utils.get_color("red").expand(B, P, 3).to(device)
            )
            contact_obj_textures = (
                mesh_utils.get_color("red").expand(B, V, 3).to(device)
            )
            contact_left = contact[..., 0].reshape(B, 1, 1).to(device)  # (B, )
            contact_right = contact[..., 1].reshape(B, 1, 1).to(device)  # (B, )
            contact_obj = ((contact_left > 0.5) | (contact_right > 0.5)).float()

            left_textures = (
                contact_left * contact_hand_textures
                + (1 - contact_left) * left_textures
            )
            right_textures = (
                contact_right * contact_hand_textures
                + (1 - contact_right) * right_textures
            )
            obj_textures = (
                contact_obj * contact_obj_textures + (1 - contact_obj) * obj_textures
            )

        oObj_verts_exp = oObj.verts_padded().repeat(B, 1, 1)
        oObj_face_exp = oObj.faces_padded().repeat(B, 1, 1)
        oObj_exp = Meshes(verts=oObj_verts_exp, faces=oObj_face_exp).to(device)
        oObj_exp.textures = mesh_utils.pad_texture(oObj_exp, obj_textures)

        left_hand_meshes.textures = mesh_utils.pad_texture(
            left_hand_meshes, left_textures
        )
        right_hand_meshes.textures = mesh_utils.pad_texture(
            right_hand_meshes, right_textures
        )

        wObj_exp = mesh_utils.apply_transform(
            oObj_exp, geom_utils.se3_to_matrix_v2(wTo)
        )
        wScene = mesh_utils.join_scene([left_hand_meshes, right_hand_meshes, wObj_exp])

        if coord:
            coord = plot_utils.create_coord(device, B, size=0.2)
            wScene = mesh_utils.join_scene([wScene, coord])
        return wScene

    def log_hoi_step_from_cam_side(
        self,
        left_hand_meshes,
        right_hand_meshes,
        wTo,
        oObj,
        contact,
        intr_pix,
        wTc,
        device,
        save_to_file=True,
        pref="training/",
        debug_info: list = None,
    ):        
        B = wTc.shape[0]
        # to ndc
        intr = intr_pix.clone()[None].repeat(B, 1, 1)
        H = (intr_pix[..., 0, 2] * 2).item()
        W = (intr_pix[..., 1, 2] * 2).item()
        intr =mesh_utils.intr_from_screen_to_ndc(intr, W, H)

        wScene = self.get_scene(
            left_hand_meshes, right_hand_meshes, wTo, oObj, contact, device, coord=False,
        )
        fxfy, pxpy = mesh_utils.get_fxfy_pxpy(intr)
        camera = PerspectiveCameras(fxfy, pxpy, device=device)
        cScene = mesh_utils.apply_transform(wScene, geom_utils.inverse_rt_v2(geom_utils.se3_to_matrix_v2(wTc)))
        image_cam = mesh_utils.render_mesh(cScene, camera, out_size=360)  # (T, 3, H, W)

        nTw = mesh_utils.get_nTw(wScene.verts_packed()[None], new_scale=1.2)
        print("render wScene", wScene.verts_padded().shape, nTw.shape)

        coord = plot_utils.create_coord(device, len(wScene), size=0.2)
        wScene = mesh_utils.join_scene([wScene, coord])

        image_side = render_scene_from_azel(wScene, nTw, az=160, el=5, out_size=360)

        image_all = torch.cat([image_cam["image"], image_side["image"]], dim=-1) 
        if save_to_file:
            fname = osp.join(self.save_dir, f"{pref}")
            image_utils.save_gif(
                image_all.unsqueeze(1), fname, text_list=debug_info, fps=30, ext=".mp4"
            )
            return fname + ".mp4"
        else:
            return image_all



    def log_hoi_step_from_cam(
        self,
        left_hand_meshes,
        right_hand_meshes,
        wTo,
        oObj,
        contact,
        intr_pix,
        wTc,
        device,
        save_to_file=True,
        pref="training/",
        debug_info: list = None,
    ):
        B = wTc.shape[0]
        # to ndc
        intr = intr_pix.clone()[None].repeat(B, 1, 1)
        H = (intr_pix[..., 0, 2] * 2).item()
        W = (intr_pix[..., 1, 2] * 2).item()
        print(H, W)
        intr =mesh_utils.intr_from_screen_to_ndc(intr, W, H)

        wScene = self.get_scene(
            left_hand_meshes, right_hand_meshes, wTo, oObj, contact, device, coord=False,
        )
        fxfy, pxpy = mesh_utils.get_fxfy_pxpy(intr)
        print(fxfy, pxpy)
        camera = PerspectiveCameras(fxfy, pxpy, device=device)
        cScene = mesh_utils.apply_transform(wScene, geom_utils.inverse_rt_v2(geom_utils.se3_to_matrix_v2(wTc)))
        image = mesh_utils.render_mesh(cScene, camera, out_size=360)  # (T, 3, H, W)

        image_list = image["image"]
        if save_to_file:
            fname = osp.join(self.save_dir, f"{pref}")
            image_utils.save_gif(
                image_list.unsqueeze(1), fname, text_list=debug_info, fps=30, ext=".mp4"
            )
            return fname + ".mp4"
        else:
            return image_list

    def log_hoi_step(
        self,
        left_hand_meshes,
        right_hand_meshes,
        wTo,
        oObj,
        pref="training/",
        step=0,
        save_to_file=True,
        device="cuda:0",
        contact=None,
        debug_info: list = None,
    ):
        wScene = self.get_scene(
            left_hand_meshes,
            right_hand_meshes,
            wTo,
            oObj,
            contact,
            device,
        )

        nTw = mesh_utils.get_nTw(wScene.verts_packed()[None], new_scale=1.2)
        print("render wScene", wScene.verts_padded().shape, nTw.shape)

        image_list = render_scene_from_azel(wScene, nTw, az=160, el=5, out_size=360)

        image_list = image_list["image"]  # (T, 3, H, W)

        if save_to_file:
            fname = osp.join(self.save_dir, f"{pref}")
            image_utils.save_gif(
                image_list.unsqueeze(1), fname, text_list=debug_info, fps=30, ext=".mp4"
            )
            return fname + ".mp4"
        else:
            return image_list

    def log_training_step(
        self,
        left_hand_meshes,
        right_hand_meshes,
        wTo_list,
        color_list,
        uid,
        step=0,
        pref="training/",
        save_to_file=True,
        device="cuda:0",
    ):
        """
        render one seq

        :param left_hand_meshes: Meshes with T batch
        :param right_hand_meshes: [(Meshes, ) ] ?
        :param wTo: [T, 4, 4]
        :param uid
        :param wTc: [T, 4, 4]
        """
        scene_points = []
        if left_hand_meshes is not None:
            left_hand_meshes.textures = mesh_utils.pad_texture(
                left_hand_meshes, "white"
            )
            scene_points.append(left_hand_meshes.verts_packed())
        if right_hand_meshes is not None:
            right_hand_meshes.textures = mesh_utils.pad_texture(
                right_hand_meshes, "blue"
            )
            scene_points.append(right_hand_meshes.verts_packed())
        # right_hand_meshes.textures = mesh_utils.pad_texture(right_hand_meshes, 'blue')
        # device = left_hand_meshes.device
        # T = len(left_hand_meshes)
        if isinstance(uid, Meshes):
            oObj = uid.to(device)
        else:
            if uid not in self.object_cache:
                pass
            oObj = self.object_cache[uid].to(device)

        oObj_verts = oObj.verts_padded()  # (1, V, 3)

        for wTo, color in zip(wTo_list, color_list):
            wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
            wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
            wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)

            wObj_verts = mesh_utils.apply_transform(oObj_verts, wTo)
            scene_points.append(wObj_verts[0])  # (V, 3)

        scene_points = torch.cat(scene_points, 0)[None]

        nTw = mesh_utils.get_nTw(scene_points, new_scale=1.2)

        wScene = []
        if left_hand_meshes is not None:
            wScene.append(left_hand_meshes)
        if right_hand_meshes is not None:
            wScene.append(right_hand_meshes)

        for wTo, color in zip(wTo_list, color_list):
            wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
            T = wTo.shape[0]
            wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
            wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)

            wObj_verts = mesh_utils.apply_transform(oObj_verts, wTo)
            wObj_faces = oObj.faces_padded().repeat(T, 1, 1)
            wObj = Meshes(verts=wObj_verts, faces=wObj_faces).to(device)
            wObj.textures = mesh_utils.pad_texture(wObj, color)
            wScene.append(wObj)

        wScene = mesh_utils.join_scene(wScene)

        coord = plot_utils.create_coord(device, T, size=0.2)
        wScene = mesh_utils.join_scene([wScene, coord])

        image_list = render_scene_from_azel(wScene, nTw, az=160, el=5, out_size=360)
        print("render done!")

        image_list = image_list["image"]

        if save_to_file:
            fname = osp.join(self.save_dir, f"{pref}{step:07d}")
            image_utils.save_gif(image_list.unsqueeze(1), fname, fps=30, ext=".mp4")
            return fname + ".mp4"
        else:
            return image_list


def render_scene_from_azel(
    wScene, nTw, az=180, el=0, degree=True, f=10, new_bound=1, r=0.8, *args, **kwargs
):
    dist = f * new_bound / r

    device = nTw.device
    N = len(wScene)
    nTw_exp = nTw.reshape(1, 4, 4).repeat(N, 1, 1)

    azel = torch.tensor([[az, el]], device=device)
    if degree:
        azel = azel * np.pi / 180

    azel = azel.repeat(N, 1)
    roll = torch.zeros([N, 1], device=device)
    angle = torch.cat([azel, roll], dim=-1)

    R = euler_angles_to_matrix(angle, "YXZ")

    tsl = torch.zeros([N, 3], device=device)  # TODO
    tsl[..., 2] = dist
    tsl = tsl.reshape(N, 3)
    cTn = geom_utils.rt_to_homo(R, tsl)

    cTw_rot, cTw_tsl = look_at_view_transform(
        dist, el, az, True, up=((0, 0, 1),), device=device
    )

    cTw = cTn @ nTw_exp

    # cTw = geom_utils.rt_to_homo(cTw_rot, cTw_tsl).repeat(N, 1, 1)
    # cTw = cTw @ nTw_exp

    cTw = cTw.reshape(N, 4, 4)
    cScene = mesh_utils.apply_transform(wScene, cTw)
    cameras = PerspectiveCameras(f, device=device)
    image = mesh_utils.render_mesh(cScene, cameras, **kwargs)
    return image


def test():
    fname = "outputs/tmp/wScene.pkl"
    with open(fname, "rb") as f:
        data = pickle.load(f)
    wScene = data["wScene"].cuda()
    nTw = data["nTw"].cuda()

    N = len(wScene)
    coord = plot_utils.create_coord("cuda", N, size=0.2)
    wScene = mesh_utils.join_scene([wScene, coord])

    image = render_scene_from_azel(wScene, nTw, az=160, el=5, out_size=500)

    wScene = plot_utils.create_coord("cuda", 1, size=0.2)
    image_utils.save_gif(
        image["image"].unsqueeze(1), "outputs/tmp/test.mp4", fps=30, ext=".mp4"
    )

    image_list = mesh_utils.render_geom_rot_v2(wScene, "az")
    image_utils.save_gif(image_list, "outputs/tmp/test_az", fps=30, ext=".mp4")

    image_list = mesh_utils.render_geom_rot_v2(wScene, "el")
    image_utils.save_gif(image_list, "outputs/tmp/test_el", fps=30, ext=".mp4")

    image_list = mesh_utils.render_geom_rot_v2(wScene, "el0")
    image_utils.save_gif(image_list, "outputs/tmp/test_el0", fps=30, ext=".mp4")


if __name__ == "__main__":
    test()
