import viser
import numpy as np
from pathlib import Path
from egorecon.visualization.pt3d_visualizer import Pt3dVisualizer



class VisorVisualizer:
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
        self.port = 8521

    @staticmethod
    def setup_template(object_mesh_dir, lib="hotclip"):
        return Pt3dVisualizer.setup_template(object_mesh_dir)


    @staticmethod
    def vis_bps(disp, obj_basis, mesh):
        """
        Visualize BPS (Basis Point Set) data with viser interface.
        
        :param disp: (B, P, 3) displacement vectors
        :param obj_basis: (B, P, 3) object basis points
        :param mesh: Meshes (B, ..., ) PyTorch3D meshes
        """
        import torch
        
        # Convert tensors to numpy if needed
        if isinstance(disp, torch.Tensor):
            disp = disp.detach().cpu().numpy()
        if isinstance(obj_basis, torch.Tensor):
            obj_basis = obj_basis.detach().cpu().numpy()
        
        B, P, _ = disp.shape
        
        # Create viser server
        server = viser.ViserServer(port=8522)  # Use different port to avoid conflicts
        print("Viser server started for BPS visualization. Open your browser to http://localhost:8080")
        
        # Configure scene
        server.scene.configure_default_lights()
        server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))
        
        # Add object basis points as blue points for first batch
        basis_points_current = obj_basis[0]  # (P, 3)
        basis_handle = server.scene.add_point_cloud(
            "/obj_basis_points",
            points=basis_points_current,
            colors=np.full((P, 3), (0, 0, 255)),  # Blue points
            point_size=0.01,
        )
        
        # Add displacement lines (red lines from obj_basis to obj_basis + disp)
        disp_lines_current = []
        for i in range(P):
            start_point = obj_basis[0][i]  # (3,)
            end_point = obj_basis[0][i] + disp[0][i]  # (3,)
            disp_lines_current.extend([start_point, end_point])
        
        disp_lines_current = np.array(disp_lines_current).reshape(-1, 2, 3)  # (P, 2, 3)
        disp_colors = np.full((P, 2, 3), (255, 0, 0))  # Red lines
        
        lines_handle = server.scene.add_line_segments(
            "/disp_lines",
            points=disp_lines_current,
            colors=disp_colors,
            line_width=3.0,
        )
        
        # Add mesh for first batch
        mesh_verts = mesh.verts_list()[0].detach().cpu().numpy()
        mesh_faces = mesh.faces_list()[0].detach().cpu().numpy()
        mesh_handle = server.scene.add_mesh_simple(
            "/mesh",
            vertices=mesh_verts,
            faces=mesh_faces,
            color=(255, 255, 255),  # White
            wireframe=False,
        )
        
        # Add GUI controls
        with server.gui.add_folder("Visualization Controls"):
            gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
            gui_show_lines = server.gui.add_checkbox("Show Displacement Lines", initial_value=True)
            gui_show_points = server.gui.add_checkbox("Show Basis Points", initial_value=True)
            gui_show_mesh = server.gui.add_checkbox("Show Mesh", initial_value=True)
        
        with server.gui.add_folder("Animation Controls"):
            gui_batch_slider = server.gui.add_slider(
                "Batch", min=0, max=B-1, step=1, initial_value=0
            )
            gui_play_auto = server.gui.add_checkbox("Auto Play", initial_value=False)
            gui_play_speed = server.gui.add_slider(
                "Play Speed", min=1, max=10, step=1, initial_value=5
            )
        
        # Function to update visualization based on batch
        def update_visualization(batch_idx):
            # Update basis points for the current batch
            basis_points_current = obj_basis[batch_idx]  # (P, 3)
            basis_handle.points = basis_points_current
            
            # Update displacement lines for the current batch
            disp_lines_current = []
            for i in range(P):
                start_point = obj_basis[batch_idx][i]  # (3,)
                end_point = obj_basis[batch_idx][i] + disp[batch_idx][i]  # (3,)
                disp_lines_current.extend([start_point, end_point])
            
            disp_lines_current = np.array(disp_lines_current).reshape(-1, 2, 3)  # (P, 2, 3)
            lines_handle.points = disp_lines_current
            
            # Update mesh for the current batch
            mesh_verts_batch = mesh.verts_list()[batch_idx].detach().cpu().numpy()
            mesh_faces_batch = mesh.faces_list()[batch_idx].detach().cpu().numpy()
            mesh_handle.vertices = mesh_verts_batch
            mesh_handle.faces = mesh_faces_batch
        
        # GUI callbacks
        @gui_wireframe.on_update
        def _(_):
            mesh_handle.wireframe = gui_wireframe.value
        
        @gui_show_lines.on_update
        def _(_):
            lines_handle.visible = gui_show_lines.value
        
        @gui_show_points.on_update
        def _(_):
            basis_handle.visible = gui_show_points.value
        
        @gui_show_mesh.on_update
        def _(_):
            mesh_handle.visible = gui_show_mesh.value
        
        @gui_batch_slider.on_update
        def _(_):
            update_visualization(gui_batch_slider.value)
        
        # Add information display
        with server.gui.add_folder("Information"):
            info_text = f"""
            <h3>BPS (Basis Point Set) Visualization</h3>
            <p><b>Total batches:</b> {B}</p>
            <p><b>Number of basis points:</b> {P}</p>
            <p><b>Mesh vertices:</b> {mesh_verts.shape[0]}</p>
            <p><b>Mesh faces:</b> {mesh_faces.shape[0]}</p>
            <hr>
            <h4>Visualization Elements:</h4>
            <p><b>Blue points:</b> Object basis points</p>
            <p><b>Red lines:</b> Displacement vectors (from basis to basis+disp)</p>
            <p><b>White mesh:</b> Object mesh</p>
            """
            server.gui.add_markdown("/info", info_text)
        
        # Keep server running
        print("Press Ctrl+C to exit")
        server.sleep_forever()



    @staticmethod
    def vis_w_visor(wObj, wHands, wNN, wJoints):
        """Visualize hand-object interaction using viser web interface.
        
        This function creates a viser server and renders the same content as the vis() function
        but in an interactive web-based 3D viewer instead of generating static images.
        
        :param wObj: PyTorch3D Meshes object containing object mesh
        :param wHands: PyTorch3D Meshes object containing hand meshes
        :param wNN: Tensor of neural network output joint positions (B, J, 3)
        :param wJoints: Tensor of joint positions (B, J, 3)
        """
        # Check tensor shapes
        if len(wNN.shape) != 3 or wNN.shape[2] != 3:
            raise ValueError(f"wNN must have shape (B, J, 3), got {wNN.shape}")
        if len(wJoints.shape) != 3 or wJoints.shape[2] != 3:
            raise ValueError(f"wJoints must have shape (B, J, 3), got {wJoints.shape}")
        
        B, J, _ = wNN.shape
        
        # Create viser server
        server = viser.ViserServer(port=8521)
        print("Viser server started. Open your browser to http://localhost:8080")
        
        # Configure scene
        server.scene.configure_default_lights()
        server.scene.add_grid("/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))
        
        # Convert PyTorch tensors to numpy arrays and move to CPU
        wNN_np = wNN.detach().cpu().numpy()
        wJoints_np = wJoints.detach().cpu().numpy()
        
        # Add object mesh for first batch
        obj_verts = wObj.verts_list()[0].detach().cpu().numpy()
        obj_faces = wObj.faces_list()[0].detach().cpu().numpy()
        object_handle = server.scene.add_mesh_simple(
            "/object",
            vertices=obj_verts,
            faces=obj_faces,
            color=(255, 255, 255),  # White
            wireframe=False,
        )
        
        # Add hand meshes for first batch
        hand_verts = wHands.verts_list()[0].detach().cpu().numpy()
        hand_faces = wHands.faces_list()[0].detach().cpu().numpy()
        hands_handle = server.scene.add_mesh_simple(
            "/hands",
            vertices=hand_verts,
            faces=hand_faces,
            color=(0, 0, 255),  # Blue
            wireframe=False,
        )
        
        # Add line segments between joints for first batch
        wNN_current = wNN_np[0]  # (J, 3)
        wJoints_current = wJoints_np[0]  # (J, 3)
        
        # Create line segments: each line goes from wNN[i] to wJoints[i]
        line_points = np.stack([wNN_current, wJoints_current], axis=1)  # (J, 2, 3)
        line_colors = np.full((line_points.shape[0], 2, 3), (255, 0, 0))  # Red lines
        
        lines_handle = server.scene.add_line_segments(
            "/joint_lines",
            points=line_points,
            colors=line_colors,
            line_width=3.0,
        )
        
        # Add GUI controls
        with server.gui.add_folder("Visualization Controls"):
            gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
            gui_show_lines = server.gui.add_checkbox("Show Joint Lines", initial_value=True)
            gui_show_hands = server.gui.add_checkbox("Show Hands", initial_value=True)
            gui_show_object = server.gui.add_checkbox("Show Object", initial_value=True)
        
        with server.gui.add_folder("Animation Controls"):
            gui_batch_slider = server.gui.add_slider(
                "Batch", min=0, max=B-1, step=1, initial_value=0
            )
            gui_play_auto = server.gui.add_checkbox("Auto Play", initial_value=False)
            gui_play_speed = server.gui.add_slider(
                "Play Speed", min=1, max=10, step=1, initial_value=5
            )
        
        # Function to update visualization based on batch
        def update_visualization(batch_idx):
            # Update object mesh for the current batch
            obj_verts_batch = wObj.verts_list()[batch_idx].detach().cpu().numpy()
            obj_faces_batch = wObj.faces_list()[batch_idx].detach().cpu().numpy()
            object_handle.vertices = obj_verts_batch
            object_handle.faces = obj_faces_batch
            
            # Update hand mesh for the current batch
            hand_verts_batch = wHands.verts_list()[batch_idx].detach().cpu().numpy()
            hand_faces_batch = wHands.faces_list()[batch_idx].detach().cpu().numpy()
            hands_handle.vertices = hand_verts_batch
            hands_handle.faces = hand_faces_batch
            
            # Update line segments for the current batch
            wNN_current = wNN_np[batch_idx]  # (J, 3)
            wJoints_current = wJoints_np[batch_idx]  # (J, 3)
            
            # Create line segments: each line goes from wNN[i] to wJoints[i]
            line_points = np.stack([wNN_current, wJoints_current], axis=1)  # (J, 2, 3)
            line_colors = np.full((line_points.shape[0], 2, 3), (255, 0, 0))  # Red lines
            
            # Update the line segments
            lines_handle.points = line_points
            lines_handle.colors = line_colors
        
        # GUI callbacks
        @gui_wireframe.on_update
        def _(_):
            object_handle.wireframe = gui_wireframe.value
            hands_handle.wireframe = gui_wireframe.value
        
        @gui_show_lines.on_update
        def _(_):
            lines_handle.visible = gui_show_lines.value
        
        @gui_show_hands.on_update
        def _(_):
            hands_handle.visible = gui_show_hands.value
        
        @gui_show_object.on_update
        def _(_):
            object_handle.visible = gui_show_object.value
        
        @gui_batch_slider.on_update
        def _(_):
            update_visualization(gui_batch_slider.value)
        
        # Add information display
        with server.gui.add_folder("Information"):
            info_text = f"""
            <h3>Hand-Object Interaction Visualization</h3>
            <p><b>Total batches:</b> {B}</p>
            <p><b>Number of joints:</b> {J}</p>
            <p><b>Object vertices:</b> {obj_verts.shape[0]}</p>
            <p><b>Hand vertices:</b> {hand_verts.shape[0]}</p>
            <p><b>Joint lines:</b> {J}</p>
            """
            server.gui.add_markdown("/info", info_text)
        
        # Keep server running
        print("Press Ctrl+C to exit")
        server.sleep_forever()        