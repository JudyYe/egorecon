from glob import glob

from pathlib import Path
import torch
import os.path as osp
import pytorch3d
from jutils import plot_utils, mesh_utils, image_utils
from pytorch3d.structures import Meshes


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
        self.save_dir = Path(save_dir) / 'log'
        self.enable_visualization = enable_visualization
        self.mano_models_dir = Path(mano_models_dir)
        self.object_mesh_dir = Path(object_mesh_dir)
        self.object_cache = {}

        self.setup_template()

    def setup_template(self, lib='hot3d'):
        if lib == 'hot3d':
            glob_file = glob(osp.join(self.object_mesh_dir, "*.glb"))
            for mesh_file in glob_file:
                uid = osp.basename(mesh_file).split('.')[0]
                self.object_cache[uid] = mesh_utils.load_mesh(mesh_file)
        else:
            raise ValueError(f"Invalid library: {lib}")

    
    def log_training_step(self, left_hand_meshes, right_hand_meshes, wTo_list, color_list, uid, step=0, pref='training/'):
        """
        render one seq
        
        :param left_hand_meshes: Meshes with T batch
        :param right_hand_meshes: [(Meshes, ) ] ? 
        :param wTo: [T, 4, 4] 
        :param uid
        :param wTc: [T, 4, 4]
        """
        T = len(left_hand_meshes)
        oObj = self.load_mesh(uid)
        oObj_verts = oObj.verts_padded()  # (1, V, 3)

        
        scene_points = [left_hand_meshes.verts_packed(), right_hand_meshes.verts_packed()]
        for wTo, color in zip(wTo_list, color_list):
        # get_boundary 
            wObj_verts = mesh_utils.apply_transform(oObj_verts, wTo)
            scene_points.append(wObj_verts[0])  # (V, 3)
        
        scene_points = torch.cat(scene_points, 0)
        
        nTw = mesh_utils.get_nTw(scene_points, new_scale=1.2)


        wScene = [left_hand_meshes, right_hand_meshes]  # bathc=T

        for wTo, color in zip(wTo_list, color_list):
            wObj = mesh_utils.apply_transform(oObj_verts, wTo)
            wObj.textures = mesh_utils.pad_texture(wObj, color)
            wScene.append(wObj)

        wScene = mesh_utils.join_scene(wScene)
        image_list = mesh_utils.render_geom_rot_v2(wScene, nTw=nTw, time_len=0, out_size=512)

        image_list = image_list[0]  # (T, 3, H, W)

        fname = osp.join(self.save_dir, f"{pref}{step:07d}")

        image_utils.save_gif(image_list.unsqueeze(1), fname, fps=30, ext='.mp4')
        return fname + '.mp4'