from glob import glob

from pathlib import Path
import torch
import os.path as osp
import pytorch3d
from jutils import plot_utils, mesh_utils, image_utils, geom_utils
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
            print('glob_file', glob_file, 'object_mesh_dir', self.object_mesh_dir)
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
        device = left_hand_meshes.device
        T = len(left_hand_meshes)
        if uid not in self.object_cache:
            print(uid, self.object_cache.keys() )
        oObj = self.object_cache[uid].to(device)
        oObj_verts = oObj.verts_padded()  # (1, V, 3)
        
        scene_points = [left_hand_meshes.verts_packed(), right_hand_meshes.verts_packed()]
        for wTo, color in zip(wTo_list, color_list):
            wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
            wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
            wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)
        
            wObj_verts = mesh_utils.apply_transform(oObj_verts, wTo)
            scene_points.append(wObj_verts[0])  # (V, 3)
        
        scene_points = torch.cat(scene_points, 0)[None]
        print('scene_points shape', scene_points.shape)
        
        nTw = mesh_utils.get_nTw(scene_points, new_scale=1.2)

        wScene = [left_hand_meshes, right_hand_meshes]  # bathc=T

        for wTo, color in zip(wTo_list, color_list):
            wTo_tsl, wTo_6d = wTo[..., :3], wTo[..., 3:]
            wTo_mat = geom_utils.rotation_6d_to_matrix(wTo_6d)
            wTo = geom_utils.rt_to_homo(wTo_mat, wTo_tsl)

            wObj_verts = mesh_utils.apply_transform(oObj_verts, wTo)
            wObj_faces = oObj.faces_padded().repeat(T, 1, 1)
            wObj = Meshes(verts=wObj_verts, faces=wObj_faces).to(device)
            wObj.textures = mesh_utils.pad_texture(wObj, color)
            wScene.append(wObj)

        wScene = mesh_utils.join_scene(wScene)
        print('wScene len', len(wScene), 'nTw', nTw.shape)

        image_list = mesh_utils.render_geom_rot_v2(wScene, nTw=nTw.repeat(T, 1, 1), time_len=1, out_size=512)

        image_list = image_list[0]  # (T, 3, H, W)

        fname = osp.join(self.save_dir, f"{pref}{step:07d}")

        image_utils.save_gif(image_list.unsqueeze(1), fname, fps=30, ext='.mp4')
        return fname + '.mp4'