import os.path as osp
import pytorch3d
from hot3d.hot3d.data_loaders.loader_object_library import load_object_library
from jutils import plot_utils

class Pt3dVisualizer:
    def __init__(self):
        pass

    def setup_template(self, lib='hot3d'):
        if lib == 'hot3d':
            root_dir = "hot3d/hot3d/dataset/"

            mesh_file = osp.join(root_dir, "assets", f"{uid}.glb")
            object_library = load_object_library(
            object_library_folderpath=object_library_folder
        )


    def save_video(self,):
        return 
    
    def log_training_step(self, wTo, wTc,  pref='training/'):
        """
        render one seq
        
        :param left_hand_meshes: (Meshes, ) ? 
        :param right_hand_meshes: (Meshes, ) ? 
        :param wTo: [T, 4, 4] 
        :param object_meshes: [T, D, 3]
        :param wTc: [T, 4, 4]
        """
        camera_meshes =plot_utils.vis_cam(wTc=wTc) 

        
        
        return 