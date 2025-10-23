import pickle
from pytorch3d.ops import sample_points_from_meshes
import os
import os.path as osp
from egorecon.visualization.visor_visualizer import VisorVisualizer
from tqdm import tqdm
import hydra
import torch
from jutils import model_utils, plot_utils, image_utils, geom_utils, mesh_utils
from move_utils.slurm_utils import slurm_engine
from omegaconf import OmegaConf
from ..manip.model import build_model
from ..manip.model.guidance_optimizer_jax import (
    do_guidance_optimization,
    se3_to_wxyz_xyz,
    wxyz_xyz_to_se3,
)
from .trainer_hoi import gen_vis_res, vis_gen_process
from .test_hoi import build_model_from_ckpt, set_test_cfg
from ..manip.model.transformer_hand_to_object_diffusion_model import (
    CondGaussianDiffusion,
)
from ..manip.data import build_dataloader
from ..utils.evaluation_metrics import compute_wTo_error
from ..visualization.pt3d_visualizer import Pt3dVisualizer
from pytorch3d.structures import Meshes
device = torch.device("cuda:0")



@hydra.main(config_path="../../config", config_name="test", version_base=None)
@slurm_engine()
def debug(opt):
    diffusion_model = build_model_from_ckpt(opt)
    set_test_cfg(opt, diffusion_model.opt)

    pkl_file = 'outputs/hoi/contactTrue_joint_obj/eval/test_guided_0001.pkl'
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    x_list = data['x_list']
    wJoints_data = data['wJoints']
    newPoints = data['newPoints']
    wTo = data['wTo']
    batch = data['batch']

    for i, (xt, t) in enumerate(x_list):
        if i >= 1:
            break
        denorm_xt = diffusion_model.denormalize_data(xt)
        out = diffusion_model.decode_dict(denorm_xt)
        wTo_pred = out['wTo']
        coord = diffusion_model.encode_hand_sensor(wJoints_data, wTo_pred, newPoints)  # o-h (B, T, J*3) # per step hand sensor feature  
        B, T, J_3 = wJoints_data.shape
        J = J_3 // 3

        wNN = coord + wJoints_data  # (B, T, J, 3)
        wNN = wNN.reshape(B*T, J, 3)
        wJoints = wJoints_data.reshape(B*T, J, 3)

        oObj = batch["newMesh"]
        oObj_exp = Meshes(verts=oObj.verts_list()[0][None].repeat(B*T, 1, 1), faces=oObj.faces_list()[0][None].repeat(B*T, 1, 1)).to(device)
        wTo_pred = geom_utils.se3_to_matrix_v2(wTo_pred).reshape(B*T, 4, 4)
        print(len(oObj_exp), wTo_pred.shape)
        wObj_exp = mesh_utils.apply_transform(oObj_exp, wTo_pred)

        wHands = diffusion_model.decode_hand_mesh(batch["left_hand_params"][0], batch["right_hand_params"][0], hand_rep="theta")
        wHands = mesh_utils.join_scene(wHands)

        print(wNN.shape, wJoints.shape)
        # image_list = vis(wObj_exp, wHands, wNN, wJoints)
        # image_utils.save_gif(image_list, f"outputs/debug_handsensor/vis_hand_cond_{i:04d}", text_list=[[f"T={t}"] * T], ext='.mp4', max_size=512*8)
        # VisorVisualizer.vis_w_visor(wObj_exp, wHands, wNN, wJoints)



def vis(wObj, wHands, wNN, wJoints):
    B, J, _ = wNN.shape
    print(wNN[0])
    import ipdb; ipdb.set_trace()
    verts, faces = plot_utils.create_line(wNN.reshape(B*J, 3), wJoints.reshape(B*J, 3))  #
    print('line', verts.shape, faces.shape, B)
    wLines = Meshes(verts=verts.reshape(B, -1, 3), faces=faces.reshape(B, -1, 3)).to(device)
    wLines.textures = mesh_utils.pad_texture(wLines, 'red')

    wObj.textures = mesh_utils.pad_texture(wObj, 'white')
    wHands.textures = mesh_utils.pad_texture(wHands, 'blue')

    wScene = mesh_utils.join_scene([wObj, wHands, wLines])

    image_list = mesh_utils.render_geom_rot_v2(wScene, )

    return image_list


@hydra.main(config_path="../../config", config_name="test", version_base=None)
def vis_bps(opt):
    obj_cache = VisorVisualizer.setup_template(opt.paths.object_mesh_dir)
    # largest object in obj_cache
    largest_extent = None
    largest_object = None
    for obj_name, obj_mesh in obj_cache.items():
        assert isinstance(obj_mesh, Meshes), f"Object {obj_name} is not a Meshes object"
        bbox = obj_mesh.get_bounding_boxes()  # N, 3, 2
        # extent = (bbox[..., 1] - bbox[..., 0]).norm(dim=-1).max()
        extent = bbox.norm(dim=-2).max()
        if largest_extent is None or extent > largest_extent:
            largest_extent = extent
            largest_object = obj_name
    print(f"Largest object: {largest_object} with extent {largest_extent}")

    # get the mesh whose mean is furthest from the origin
    
    # max_dist = 0
    # for obj_name, obj_mesh in obj_cache.items():
    #     obj_mean = obj_mesh.verts_list()[0].mean(dim=0)
    #     dist = obj_mean.norm()
    #     if dist > max_dist:
    #         max_dist = dist
    #         largest_object = obj_name

    newPoints = sample_points_from_meshes(obj_cache[largest_object], 5000)
    newMesh = obj_cache[largest_object]
    
    
    diffusion_model = build_model_from_ckpt(opt)
    set_test_cfg(opt, diffusion_model.opt)

    # pkl_file = 'outputs/hoi/contactTrue_joint_obj/eval/test_guided_0001.pkl'
    # with open(pkl_file, 'rb') as f:
    #     data = pickle.load(f)

    # x_list = data['x_list']
    # wJoints_data = data['wJoints']
    # newPoints = data['newPoints']
    # wTo = data['wTo']
    # batch = data['batch']
    # newMesh = batch["newMesh"]

    radius = 0.25 * 2
    # radius = 0.22 * 2
    diffusion_model.bps['obj'] *= radius / 2
    diffusion_model.obj_bps = diffusion_model.bps['obj']

    bps = diffusion_model.bps['obj'].to(newPoints.device)
    newCom = 0
    # newCom = newPoints.mean(dim=1)
    delta = diffusion_model.encode_bps(newPoints)

    print(delta.shape, (newCom + bps).shape, len(newMesh))
    print("delta (displacement):", delta.shape)
    print("obj_basis (newCom + bps):", (newCom + bps).shape)
    print("newMesh:", len(newMesh))
    VisorVisualizer.vis_bps(delta, (newCom + bps), newMesh)

    # # The shapes don't match - delta has 42 points, obj_basis has 5000 points
    # # Let's use the smaller dimension for both
    # min_points = min(delta.shape[1], (newCom + bps).shape[1])
    # print(f"Using {min_points} points for visualization")
    
    # VisorVisualizer.vis_bps(delta[:, :min_points], (newCom + bps)[:, :min_points], newMesh)
    return 
if __name__ == "__main__":
    vis_bps()