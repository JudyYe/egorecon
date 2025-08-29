from collections import defaultdict
import os.path as osp
import pickle
import sys

import cv2
from tqdm import tqdm

# sys.path.insert(0, os.path.dirname(__file__) + '../Metric3D')
# from metric import Metric3D
import imageio
import numpy as np
import nvdiffrast.torch as dr
import PIL
import torch
import trimesh
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from hot3d.hot3d.data_loaders.loader_object_library import load_object_library

# from FoundationPose.estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Metric3D.metric3d_wrapper import Metric3D
from hot3d.hot3d.dataset_api import Hot3dDataProvider
# model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True)
# rgb = imageio.imread('Metric3D/data/kitti_demo/rgb/0000000100.png')
# rgb = ToTensor()(rgb)
# pred_depth, confidence, output_dict = model.inference({'input': rgb[None]})
# print(pred_depth.shape, rgb.shape)


class PoseWrapper:
    def __init__(self, mesh_file, debug=False):
        self.depth_model = Metric3D("Metric3D/weight/metric_depth_vit_large_800k.pth")
        mesh = trimesh.load(mesh_file)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.pose_model = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir="outputs/",
            debug=debug,
            glctx=glctx,
        )
        self.mesh = mesh
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        self.to_origin = to_origin
        self.bbox = bbox

        self.need_init = True
        self.intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]

    def predict_depth(self, rgb, intrinsic=None):
        if isinstance(rgb, str):
            rgb = cv2.imread(rgb)[..., ::-1]
        elif isinstance(rgb, PIL.Image.Image):
            rgb = np.array(rgb)
        else:
            rgb = rgb
        H, W = rgb.shape[:2]

        if intrinsic is None:
            intrinsic = self.intrinsic
        pred_depth = self.depth_model(rgb, intrinsic)
        pred_depth = cv2.resize(pred_depth, (W, H))
        return pred_depth, intrinsic

    def track_one(self, rgb, intrinsic=None):
        depth, intrinsic = self.predict_depth(rgb, intrinsic)
        fx, fy, cx, cy = intrinsic
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        K = K.reshape(3, 3)

        pose = self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=2)
        return pose

    def register_one(self, rgb, intrinsic=None):
        # sam mask
        depth, intrinsic = self.predict_depth(rgb, intrinsic)
        fx, fy, cx, cy = intrinsic
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        pose = self.pose_model.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
        return pose

    def track_video(self, image_list):
        for img in image_list:
            if self.need_init:
                pose = self.register_one(img, intrinsic)
                self.need_init = False
            else:
                pose = self.track_one(img, intrinsic)
            print('pose', pose.shape, type(pose))
            yield pose
        return



class HOT3DLoader:
    def __init__(self, seq_path, object_library_folder, mano_hand_model=None, fail_on_missing_data=False):
        object_library = load_object_library(
            object_library_folderpath=object_library_folder
        )
        self.data_provider = hot3d_data_provider = Hot3dDataProvider(
            sequence_folder=seq_path,
            object_library=object_library,
            mano_hand_model=mano_hand_model,
            fail_on_missing_data=fail_on_missing_data,
        )
        self.image_stream_ids = self.data_provider.device_data_provider.get_image_stream_ids()

        self._object_pose_data_provider = hot3d_data_provider.object_pose_data_provider
        self._object_box2d_data_provider = (
            hot3d_data_provider.object_box2d_data_provider
        )
        # Object library
        self._object_library = hot3d_data_provider.object_library
        self._device_data_provider = hot3d_data_provider.device_data_provider
        self._device_pose_provider = hot3d_data_provider.device_pose_data_provider

    
    def extract_all(self):
        # save T_w_o, T_w_c
        # get intrinsics
        [extrinsics, intrinsics] = (
            self._device_data_provider.get_camera_calibration(self.image_stream_ids)
        )
        resolution=[
            intrinsics.get_image_size()[0],
            intrinsics.get_image_size()[1],
        ] 
        focal_length=float(intrinsics.get_focal_lengths()[0])
        cx, cy = resolution[0] / 2, resolution[1] / 2
        intrinsics = [focal_length, focal_length, cx, cy]
        print(intrinsics)

        # save all images 
        to_save = defaultdict(list)
        
        timestamps = self.data_provider.device_data_provider.get_sequence_timestamps()
        for timestamp_ns in timestamps:
            image_data = self._device_data_provider.get_undistorted_image(
                timestamp_ns, self.image_stream_ids
            )
            image = image_data.image

            to_save['image'].append(image)
            box2d_collection = (
                    self._object_box2d_data_provider.get_bbox_at_timestamp(
                        stream_id=self.image_stream_ids,
                        timestamp_ns=timestamp_ns,
                        time_query_options=TimeQueryOptions.CLOSEST,
                        time_domain=TimeDomain.TIME_CODE,
                    )
                ).box2d_collection  # uid2objectBox2D
            
            def box2d_to_xyxy(box2d):
                return [box2d.left, box2d.top, box2d.right, box2d.bottom]
            xyxy = {uid: box2d_to_xyxy(box2d.box2d) for uid, box2d in box2d_collection.box2ds.items()}
            to_save['box_xyxy'].append(xyxy)

            # T_w_c: T_world_device 
            T_w_c = self._device_pose_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            T_w_c = T_w_c.pose3d.T_world_device
            to_save['T_w_c'].append(T_w_c)

            T_w_o_collection = self._object_pose_data_provider.get_pose_at_timestamp(
                timestamp_ns=timestamp_ns,
                time_query_options=TimeQueryOptions.CLOSEST,
                time_domain=TimeDomain.TIME_CODE,
            )
            T_w_o_collection = T_w_o_collection.pose3d_collection.poses  # dict of uid to T_world_object
            T_w_o = {uid: T_w_o_collection[uid].T_world_object for uid in T_w_o_collection.keys()}
            to_save['T_w_o'].append(T_w_o)

        # save a intrincis 
        to_save['intrinsic'] = intrinsics
        # save a list of bbox
        return to_save

    
    

# metric_wrapper = Metric3D("Metric3D/weight/metric_depth_vit_large_800k.pth")
# intrinsic = [707.0493, 707.0493, 604.0814, 180.5066]
# pred_depth = metric_wrapper("Metric3D/data/kitti_demo/rgb/0000000100.png", intrinsic)
# img = imageio.imread("Metric3D/data/kitti_demo/rgb/0000000100.png")
# H, W = img.shape[:2]
# pred_depth = cv2.resize(pred_depth, (W, H))

# print(img.shape, pred_depth.shape)
# # rgb = imageio.imread('Metric3D/data/kitti_demo/rgb/0000000100.png')
# # pred_depth, confidence, output_dict = model.inference({'input': rgb})



def main(num=-1):
    root_data = 'hot3d/hot3d/dataset'
    seq = osp.join(root_data, 'P0003_c701bd11/')
    object_library_folder = osp.join(root_data, 'assets/')
    loader = HOT3DLoader(seq, object_library_folder, )
    to_save = loader.extract_all()
    save_dir = osp.join(root_data, 'pred_pose/P0003_c701bd11/')

    if num > 0:
        to_save['image'] = to_save['image'][:num]
    for i in tqdm(range(len(to_save['image']))):
        cv2.imwrite(f'{save_dir}/{i:05d}.jpg', to_save['image'][i])
    
    to_save.pop('image')

    # save intrincis
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/intrinsic.pkl', 'wb') as f:
        pickle.dump(to_save['intrinsic'], f)
    to_save.pop('intrinsic')

    # save box_xyxy
    with open(f'{save_dir}/box_xyxy.pkl', 'wb') as f:
        pickle.dump(to_save['box_xyxy'], f)
    to_save.pop('box_xyxy')

    # save the rest
    with open(f'{save_dir}/meta.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    


if __name__ == '__main__':
    main()