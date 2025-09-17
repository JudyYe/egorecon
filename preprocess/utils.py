
def set_device():
    global device
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    Also handles texture extraction and ensures consistent mesh properties.
    """
    import trimesh

    if isinstance(scene_or_mesh, trimesh.Scene):
        # trimesh.Scene.dump() will apply all transforms and merge geometry
        mesh = trimesh.util.concatenate(
            [g for g in scene_or_mesh.dump(concatenate=False)]
        )
    else:
        mesh = scene_or_mesh

    # Ensure mesh has vertex normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.compute_vertex_normals()

    # Handle texture extraction for PBR materials
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        material = mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            # Extract texture from PBR material
            try:
                texture_img = np.array(material.baseColorTexture.convert('RGB'))
                # Store texture for later use if needed
                mesh._texture_image = texture_img
            except Exception as e:
                print(f"Warning: Could not extract texture: {e}")
                mesh._texture_image = None
        elif hasattr(material, 'baseColorFactor'):
            # Use solid color from baseColorFactor
            color = material.baseColorFactor[:3]  # RGB
            mesh._texture_image = np.full((512, 512, 3), color, dtype=np.uint8)
        else:
            mesh._texture_image = None
    else:
        mesh._texture_image = None

    # Ensure vertex colors are available if possible
    if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
        # Create default vertex colors if none exist
        if hasattr(mesh, '_texture_image') and mesh._texture_image is not None:
            # Use average color from texture
            avg_color = np.mean(mesh._texture_image.reshape(-1, 3), axis=0)
            mesh.visual.vertex_colors = np.tile(avg_color, (len(mesh.vertices), 1))
        else:
            # Default gray color
            mesh.visual.vertex_colors = np.full((len(mesh.vertices), 3), [128, 128, 128], dtype=np.uint8)

    return mesh

class UniDepthWrapper():
    def __init__(self):
        version = "v2"
        backbone = "vitl14"
        self.depth_model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True)
        self.depth_model.eval()
        self.depth_model.to(device)

    def unproject(self, mask, intrinsic, depth):
        fx, fy, cx, cy = intrinsic
        intrinsic = torch.FloatTensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        depth = torch.from_numpy(depth).permute(2, 0, 1)

        # unproject depth map to points in camera frame
        
    def __call__(self, rgb, intrinsic):
        fx, fy, cx, cy = intrinsic
        intrinsic = torch.FloatTensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # from unidepth.utils.camera import Pinhole
        # print(intrinsics.shape)
        rgb = torch.from_numpy(rgb/255.0).permute(2, 0, 1) # C, H, W

        # prediction = self.depth_model.infer(rgb, Pinhole(K=intrinsics))
        prediction = self.depth_model.infer(rgb, intrinsic)
        depth = prediction["depth"]

        depth = depth.squeeze().cpu().numpy().copy()
        return depth, prediction
