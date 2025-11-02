from .transformer_hand_to_object_diffusion_model import CondGaussianDiffusion


def build_model(opt) -> CondGaussianDiffusion:

    # Define model

    hand_rep = opt.get('hand_rep', 'joint')
    if hand_rep == 'theta':
        hand_rep_dim = (3 + 3 + 15 + 10) * 2  # (2 X (3 + 3 + 15 + 10))
    elif hand_rep == 'joint':
        hand_rep_dim = 21 * 3 * 2  # (2 X J X 3)
    elif hand_rep == 'joint_theta':
        hand_rep_dim = 21 * 3 * 2 + (3 + 3 + 15 + 10) * 2  # (2 X J X 3 + 2 X (3 + 3 + 15 + 10))
    else:
        raise ValueError(f"Invalid hand representation: {hand_rep}")

    hand = opt.get('hand', 'cond')
    if hand == 'cond':
        cond_dim = hand_rep_dim
    elif hand == 'out':
        cond_dim = 0
    elif hand == 'cond_out':
        cond_dim = hand_rep_dim
    else:
        raise ValueError(f"Invalid hand condition: {hand}")

    if opt.condition.noisy_obj:
        cond_dim += 9  # Input dimension (2 hands × pose_dim each)
    if opt.condition.bps == 2:
        cond_dim += 2 * 21 * 3
    else:
        pass

    if opt.condition.get("first_wTo", False):
        cond_dim += 9

    repr_dim = 9  # Output dimension (3D translation + 6D rotation)
    if hand == 'out':
        repr_dim += hand_rep_dim
    if hand == 'cond_out':
        repr_dim += hand_rep_dim
    
    if 'output' in opt and opt.output.contact:
        repr_dim += 2

    diffusion_model = CondGaussianDiffusion(
        opt,
        d_feats=repr_dim,
        out_dim=repr_dim,
        condition_dim=cond_dim,
        max_timesteps=opt.model.window + 2,
        timesteps=1000,
        loss_type="l1",
        **opt.model,
    )
    return diffusion_model