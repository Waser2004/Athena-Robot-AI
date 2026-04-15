import json
import time
import random
import socket
import struct
from pathlib import Path
import numpy as np
from math import sqrt, radians, pi, degrees

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector 
from mathutils.bvhtree import BVHTree


class RobotEnv:
    def __init__(self, fps: int=30):
        # env params
        self.fps = fps

        # action params
        self.current_velocites = [0, 0, 0, 0, 0, 0]
        self.grapper_state = False  # True closed, False open

        # define Objects
        # Robot
        self.robot_objects = {
            "base": bpy.data.objects["Robot Base"],
            "primary arm": bpy.data.objects["primary Arm"],
            "secondary arm - part 1": bpy.data.objects["secondary Arm part 1"],
            "secondary arm - part 2": bpy.data.objects["secondary Arm part 2"],
            "tertiary arm - part 1": bpy.data.objects["tertiary Arm part 1"],
            "tertiary arm - part 2": bpy.data.objects["tertiary Arm part 2"],
            "finger - right": bpy.data.objects["Finger (right)"],
            "finger - left": bpy.data.objects["Finger (left)"],

            "Grab position": bpy.data.objects["Grab position"]
        }
        
        self.camera = bpy.data.objects["OV2640"]

        # Workplate & Target Cube
        self.workplate = bpy.data.objects["Workplate"]
        self.kallax_regal = bpy.data.objects["Kallax Regal"]
        self.target_cube = bpy.data.objects["Target Cube"]

        # lighting
        self.light_fenster = bpy.data.objects["Fenster"]
        self.light_decke = bpy.data.objects["Decke"]
        self.light_sessel = bpy.data.objects["Sessel"]

        # Cache frequently reused data / nodes for long generation runs.
        self._workplate_grid_points: np.ndarray | None = None
        self._workplate_grid_file = Path(__file__).resolve().parent / "docs" / "grid_centers.txt"
        self._image_capture_nodes: dict[str, bpy.types.Node] = {}
        self._captured_frame_count = 0
    
    def set_robot_pose(self, actuator_rotations: list[float]):
        """This function sets the robot to a defined pose based on the actuator rotations"""
        self.robot_objects["base"].rotation_euler.z = actuator_rotations[0]
        self.robot_objects["primary arm"].rotation_euler.x = actuator_rotations[1]
        self.robot_objects["secondary arm - part 1"].rotation_euler.x = actuator_rotations[2]
        self.robot_objects["secondary arm - part 2"].rotation_euler.z = actuator_rotations[3]
        self.robot_objects["tertiary arm - part 1"].rotation_euler.x = actuator_rotations[4]
        self.robot_objects["tertiary arm - part 2"].rotation_euler.y = actuator_rotations[5]
        self.robot_objects["finger - right"].rotation_euler.z = pi
        self.robot_objects["finger - left"].rotation_euler.z = pi

    def set_cube_pose(self, x: float, y: float, z: float = 0.025, yaw: float | None = None):
        """Set target cube pose on the workplate."""
        self.target_cube.rotation_euler.x = 0
        self.target_cube.rotation_euler.y = 0
        self.target_cube.rotation_euler.z = random.uniform(0, 2 * pi) if yaw is None else float(yaw)

        self.target_cube.location.x = float(x)
        self.target_cube.location.y = float(y)
        self.target_cube.location.z = float(z)
    
    def reset(self, cube_position: str = "home", robot_pose: str = "home", actuator_rotations: list[float] | None = None):
        """
            This function resets the Environment to a defined starting state.
        
            Args:
                cube_position: ["home", "random_on_workplate", "random_not_on_workplate"] defines the position of the cube
                robot_pose: ["home", "resting", "random"] defines the pose of the robot
        """
        assert cube_position in ["home", "random_on_workplate", "random_not_on_workplate"], "The cube position has to be \"home\", \"random_on_workplate\" or \"random_not_on_workplate\""
        assert robot_pose in ["home", "resting", "random"], "The robot_pose has to be \"home\", \"resting\" or \"random\""

        # cube position
        if cube_position == "home":
            self.target_cube.location.x = - 0.24
            self.target_cube.location.y = 0
            self.target_cube.location.z = 0.025

        elif cube_position == "random_on_workplate":
            self.target_cube.rotation_euler.x = 0
            self.target_cube.rotation_euler.y = 0
            self.target_cube.rotation_euler.z = random.uniform(0, 6.28)

            self.target_cube.location.y = y = - random.uniform(100, 615) / 1000
            self.target_cube.location.x = - (random.uniform(sqrt(135**2 - (-y * 1000 - 25)**2) + 25, 240) if -y < 0.16 else random.uniform(-50, 240)) / 1000
            self.target_cube.location.z = 0.025

        elif cube_position == "random_not_on_workplate":
            pass

        # robot pose
        if robot_pose == "home":
            self.robot_objects["base"].rotation_euler = Euler((0, 0, -pi/2), 'XYZ')
            self.robot_objects["primary arm"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["secondary arm - part 1"].rotation_euler = Euler((-0.8185838538, 0, 0), 'XYZ')
            self.robot_objects["secondary arm - part 2"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["tertiary arm - part 1"].rotation_euler = Euler((2.389373199, 0, 0), 'XYZ')
            self.robot_objects["tertiary arm - part 2"].rotation_euler = Euler((0, pi / 2, 0), 'XYZ')
            self.robot_objects["finger - right"].rotation_euler = Euler((0, 0, pi), 'XYZ')
            self.robot_objects["finger - left"].rotation_euler = Euler((0, 0, pi), 'XYZ')

        elif robot_pose == "resting":
            self.robot_objects["base"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["primary arm"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["secondary arm - part 1"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["secondary arm - part 2"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["tertiary arm - part 1"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["tertiary arm - part 2"].rotation_euler = Euler((0, 0, 0), 'XYZ')
            self.robot_objects["finger - right"].rotation_euler = Euler((0, 0, pi), 'XYZ')
            self.robot_objects["finger - left"].rotation_euler = Euler((0, 0, pi), 'XYZ')
            
        elif robot_pose == "random":
            pass
            
        # set lighting
        self.light_fenster.data.energy = random.uniform(5, 50)
        self.light_decke.data.energy = random.uniform(5, 50)
        self.light_sessel.data.energy = random.uniform(5, 50)

        self.light_fenster.data.color = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
        self.light_decke.data.color = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
        self.light_sessel.data.color = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1))
    
    def target_cube_in_view(self) -> float:
        """
        Returns a normalized distance in [0, 1] between the cube's projected 2D center
        and the image center. 0 means perfectly centered, 1 means no vertex is visible.
        """
        bpy.context.view_layer.update()  # Force update of transforms
        scene = bpy.context.scene
        mesh = self.target_cube.data
        world_matrix = self.target_cube.matrix_world

        visible_coords = []
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            co_ndc = world_to_camera_view(scene, self.camera, world_coord)

            if (0.0 <= co_ndc.x <= 1.0 and
                0.0 <= co_ndc.y <= 1.0 and
                co_ndc.z > 0):
                visible_coords.append((co_ndc.x, co_ndc.y))

        if not visible_coords:
            return 1.0

        xs, ys = zip(*visible_coords)
        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2

        distance = sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) / sqrt(2)
        return min(distance, 1.0)

    def target_cube_within_padding(self, padding: float = 0.1) -> bool:
        """
        Returns True if any part of the projected cube overlaps the padded
        camera region [padding, 1-padding] in both x and y.
        """
        bpy.context.view_layer.update()  # Force update of transforms
        scene = bpy.context.scene
        mesh = self.target_cube.data
        world_matrix = self.target_cube.matrix_world

        # Consider only vertices in front of the camera.
        front_vertices: list[tuple[float, float]] = []
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            co_ndc = world_to_camera_view(scene, self.camera, world_coord)
            if co_ndc.z > 0:
                front_vertices.append((float(co_ndc.x), float(co_ndc.y)))

        if not front_vertices:
            return False

        xs, ys = zip(*front_vertices)
        cube_min_x, cube_max_x = min(xs), max(xs)
        cube_min_y, cube_max_y = min(ys), max(ys)

        inner_min_x, inner_max_x = 0.0 + padding, 1.0 - padding
        inner_min_y, inner_max_y = 0.0 + padding, 1.0 - padding

        overlaps_x = max(cube_min_x, inner_min_x) <= min(cube_max_x, inner_max_x)
        overlaps_y = max(cube_min_y, inner_min_y) <= min(cube_max_y, inner_max_y)
        return overlaps_x and overlaps_y

    def cube_visibility_labels(self) -> dict[str, float | bool | str]:
        """
        Calculate continuous cube-visibility labels in normalized image coordinates.

        Mirrors the historical cube-detection generator logic so filename labels
        remain compatible with existing training data parsing.
        """
        bpy.context.view_layer.update()  # Force update of transforms
        scene = bpy.context.scene
        mesh = self.target_cube.data
        world_matrix = self.target_cube.matrix_world

        projected_vertices: list[tuple[float, float, float]] = []
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            co_ndc = world_to_camera_view(scene, self.camera, world_coord)
            projected_vertices.append((float(co_ndc.x), float(co_ndc.y), float(co_ndc.z)))

        front_vertices = [(x, y) for x, y, z in projected_vertices if z > 0.0]
        if not front_vertices:
            return {
                "cube_visible": False,
                "label": "no_cube_visible",
                "visible_image_ratio": 0.0,
                "inframe_fraction": 0.0,
                "edge_margin": -1.0,
            }

        xs = [x for x, _ in front_vertices]
        ys = [y for _, y in front_vertices]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        raw_w = max(max_x - min_x, 0.0)
        raw_h = max(max_y - min_y, 0.0)
        raw_area = raw_w * raw_h

        clip_min_x = max(min_x, 0.0)
        clip_max_x = min(max_x, 1.0)
        clip_min_y = max(min_y, 0.0)
        clip_max_y = min(max_y, 1.0)

        clipped_w = max(clip_max_x - clip_min_x, 0.0)
        clipped_h = max(clip_max_y - clip_min_y, 0.0)
        clipped_area = clipped_w * clipped_h

        visible_image_ratio = min(max(clipped_area, 0.0), 1.0)
        if raw_area > 1e-12:
            inframe_fraction = min(max(clipped_area / raw_area, 0.0), 1.0)
        else:
            inframe_fraction = 1.0 if clipped_area > 0.0 else 0.0

        edge_margin = min(min_x, 1.0 - max_x, min_y, 1.0 - max_y)
        cube_visible = clipped_area > 0.0
        if not cube_visible:
            cube_visible = any(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in front_vertices)

        return {
            "cube_visible": bool(cube_visible),
            "label": "cube_visible" if cube_visible else "no_cube_visible",
            "visible_image_ratio": float(visible_image_ratio),
            "inframe_fraction": float(inframe_fraction),
            "edge_margin": float(edge_margin),
        }

    def get_state(
            self, 
            actuator_rotations:  bool = True,
            actuator_velocities: bool = True,
            target_cube_state:   bool = True,
            graper:              bool = True,
            collisions:          bool = True,
            workplate_coverage:  bool = True,
            distance_to_target:  bool = True,
            image:               bool = False,
        ):
        """Returns the state of the environment at the current time"""
        return_values = {}

        # actuator rotations
        actual_actuator_rotations = [
            self.robot_objects["base"].rotation_euler.z,
            self.robot_objects["primary arm"].rotation_euler.x,
            self.robot_objects["secondary arm - part 1"].rotation_euler.x,
            self.robot_objects["secondary arm - part 2"].rotation_euler.z,
            self.robot_objects["tertiary arm - part 1"].rotation_euler.x,
            self.robot_objects["tertiary arm - part 2"].rotation_euler.y
        ]
        if actuator_rotations:
            return_values.update({"actuator_rotations": actual_actuator_rotations})

        # actuator velocities
        if actuator_velocities:
            return_values.update({"actuator_velocities": self.current_velocites})

        # target cube state
        if target_cube_state:
            target_cube_location = self.target_cube.matrix_world.translation
            target_cube_rotation = self.target_cube.matrix_world.to_euler('XYZ')
            return_values.update({
                "target_cube_location": [target_cube_location.x, target_cube_location.y, target_cube_location.z],
                "target_cube_rotation": [target_cube_rotation.x, target_cube_rotation.y, target_cube_rotation.z]
            })

        # graper
        if graper:
            return_values.update({"graper": self.grapper_state})

        # collsion
        if collisions:
            return_values.update({"collisions": self._check_for_collision() or self._check_for_over_rotation(actual_actuator_rotations)})

        # workplate coverage
        if workplate_coverage:
            return_values.update({"workplate_coverage": self._check_point_visibility()})

        # distance to target and relative rotation
        if distance_to_target:
            loc1 = self.robot_objects["Grab position"].matrix_world.translation
            loc2 = self.target_cube.matrix_world.translation
            return_values.update({"distance_to_target": (loc1 - loc2).length})

            # relative rotation (roll, pitch, yaw) in radians
            rel_rot = self._get_relative_rotation_euler()
            return_values.update({"relative_rotation": rel_rot})
        
        # image
        if image:
            scene = bpy.context.scene
            _, _, viewer_node = self._ensure_grayscale_capture_nodes(scene)

            # Render
            bpy.ops.render.render(write_still=False)

            # Access pixel data from Viewer Node
            viewer_image = getattr(viewer_node, "image", None)
            if viewer_image is None:
                viewer_image = bpy.data.images.get("Viewer Node")
            if viewer_image is None:
                raise RuntimeError("Viewer Node image buffer is unavailable after rendering.")

            w, h = int(viewer_image.size[0]), int(viewer_image.size[1])
            if w <= 0 or h <= 0:
                w, h = scene.render.resolution_x, scene.render.resolution_y
            flat_pixels = np.empty(w * h * 4, dtype=np.float32)
            viewer_image.pixels.foreach_get(flat_pixels)
            arr = flat_pixels.reshape((h, w, 4))[:, :, 0]  # Grayscale channel
            return_values.update({"image": arr.tolist()})

            self._captured_frame_count += 1
            if self._captured_frame_count % 200 == 0:
                self._cleanup_orphan_viewer_images()

        return return_values
    
    def _get_relative_rotation_euler(self):
        """
        Returns the relative rotation (Euler angles) between the end-effector (Grab position) and the cube, in world space.
        Returns (x, y, z) in radians.
        """
        grab_obj = self.robot_objects["Grab position"]
        cube_obj = self.target_cube

        # Get world rotations
        grab_world_euler = grab_obj.matrix_world.to_euler('XYZ')
        cube_world_euler = cube_obj.matrix_world.to_euler('XYZ')

        # Relative rotation: end-effector minus cube
        rel_x = grab_world_euler.x
        rel_y = grab_world_euler.z
        rel_z = (grab_world_euler.y - cube_world_euler.z)

        return (rel_x, rel_y, rel_z)

    def _check_point_visibility(self):
        """Checks which points of the workplate are currently visible to the camera"""
        # load platepoints from file
        points_np = self._get_workplate_grid_points()

        scene = bpy.context.scene
        visibility_status = []

        for point_coords in points_np:
            # Convert NumPy row back to a mathutils.Vector
            point_world = Vector(point_coords)
            
            # Get the point's coordinates in the camera's normalized device coordinates (NDC)
            co_ndc = world_to_camera_view(scene, self.camera, point_world)
            
            # Check if the point is within the camera's view frustum
            is_visible = (0.0 <= co_ndc.x <= 1.0 and
                        0.0 <= co_ndc.y <= 1.0 and
                        co_ndc.z > 0)
                        
            visibility_status.append(is_visible)
            
        return visibility_status

    def _get_workplate_grid_points(self) -> np.ndarray:
        """Load workplate points once and reuse them for subsequent visibility checks."""
        if self._workplate_grid_points is None:
            if not self._workplate_grid_file.exists():
                raise FileNotFoundError(f"Workplate grid file not found: {self._workplate_grid_file}")
            loaded = np.loadtxt(self._workplate_grid_file)
            self._workplate_grid_points = np.atleast_2d(loaded)
        return self._workplate_grid_points

    @staticmethod
    def _ensure_input_link(links, source_socket, target_socket):
        """Ensure a compositor input socket has exactly one desired upstream link."""
        current_links = list(target_socket.links)
        if len(current_links) == 1 and current_links[0].from_socket == source_socket:
            return
        for link in current_links:
            links.remove(link)
        links.new(source_socket, target_socket)

    def _ensure_grayscale_capture_nodes(self, scene):
        """
        Ensure grayscale compositor nodes exist and are wired once.
        Reusing nodes avoids per-frame node-tree rebuild overhead.
        """
        scene.use_nodes = True
        tree = scene.node_tree
        nodes = tree.nodes
        links = tree.links

        rl = nodes.get("RobotEnv_RenderLayers")
        if rl is None or rl.bl_idname != "CompositorNodeRLayers":
            if rl is not None:
                nodes.remove(rl)
            rl = nodes.new("CompositorNodeRLayers")
            rl.name = "RobotEnv_RenderLayers"
            rl.label = "RobotEnv_RenderLayers"
            rl.location = (185, 285)

        rgb2bw = nodes.get("RobotEnv_RGBToBW")
        if rgb2bw is None or rgb2bw.bl_idname != "CompositorNodeRGBToBW":
            if rgb2bw is not None:
                nodes.remove(rgb2bw)
            rgb2bw = nodes.new("CompositorNodeRGBToBW")
            rgb2bw.name = "RobotEnv_RGBToBW"
            rgb2bw.label = "RobotEnv_RGBToBW"
            rgb2bw.location = (400, 285)

        viewer = nodes.get("RobotEnv_Viewer")
        if viewer is None or viewer.bl_idname != "CompositorNodeViewer":
            if viewer is not None:
                nodes.remove(viewer)
            viewer = nodes.new("CompositorNodeViewer")
            viewer.name = "RobotEnv_Viewer"
            viewer.label = "RobotEnv_Viewer"
            viewer.location = (750, 210)
        viewer.use_alpha = False

        self._ensure_input_link(links, rl.outputs["Image"], rgb2bw.inputs["Image"])
        self._ensure_input_link(links, rgb2bw.outputs["Val"], viewer.inputs["Image"])
        self._image_capture_nodes = {"render_layers": rl, "rgb2bw": rgb2bw, "viewer": viewer}
        return rl, rgb2bw, viewer

    @staticmethod
    def _cleanup_orphan_viewer_images():
        """
        Remove stale Viewer Node images left behind by previous node recreation.
        """
        for image in list(bpy.data.images):
            if image.users == 0 and image.name.startswith("Viewer Node."):
                bpy.data.images.remove(image)
    
    def _check_for_over_rotation(self, actuator_rotations: list[float]) -> bool:
        """This functino checks wether the joint angles are within their specified bounds"""
        rotation_bounds = [
            [-2.0943951023931953, 2.0943951023931953],
            [-1.5707963267948966, 1.5707963267948966],
            [-3.141592653589793, 0.0],
            [-1.5707963267948966, 1.5707963267948966],
            [0.0, 3.141592653589793],
            [-1.5707963267948966, 1.5707963267948966],
        ]

        return any(not (bound[0] <= rot <= bound[1]) for bound, rot in zip(rotation_bounds, actuator_rotations))
        
    def _check_for_collision(self) -> bool:
        """This function checks wehter the robot is colliding wiht itself or the workplate"""
        pairs = [
            # primary arm
            [self.robot_objects["primary arm"], self.robot_objects["secondary arm - part 2"]],
            [self.robot_objects["primary arm"], self.robot_objects["tertiary arm - part 1"]],
            [self.robot_objects["primary arm"], self.robot_objects["tertiary arm - part 2"]],
            [self.robot_objects["primary arm"], self.robot_objects["finger - right"]],
            [self.robot_objects["primary arm"], self.robot_objects["finger - left"]],
            # secondary arm
            [self.robot_objects["secondary arm - part 1"], self.workplate],
            [self.robot_objects["secondary arm - part 1"], self.target_cube],
            [self.robot_objects["secondary arm - part 2"], self.workplate],
            [self.robot_objects["secondary arm - part 2"], self.target_cube],
            [self.robot_objects["secondary arm - part 2"], self.robot_objects["base"]],
            # tertiary arm
            [self.robot_objects["tertiary arm - part 1"], self.workplate],
            [self.robot_objects["tertiary arm - part 1"], self.robot_objects["base"]],
            [self.robot_objects["tertiary arm - part 1"], self.target_cube],
            [self.robot_objects["tertiary arm - part 2"], self.workplate],
            [self.robot_objects["tertiary arm - part 2"], self.kallax_regal],
            [self.robot_objects["tertiary arm - part 2"], self.robot_objects["base"]],
            [self.robot_objects["tertiary arm - part 2"], self.target_cube],
            # fingers
            [self.robot_objects["finger - left"], self.workplate],
            [self.robot_objects["finger - left"], self.kallax_regal],
            [self.robot_objects["finger - left"], self.robot_objects["base"]],
            [self.robot_objects["finger - left"], self.target_cube],
            [self.robot_objects["finger - right"], self.workplate],
            [self.robot_objects["finger - right"], self.kallax_regal],
            [self.robot_objects["finger - right"], self.robot_objects["base"]],
            [self.robot_objects["finger - right"], self.target_cube],
        ]
        
        # do collision detection
        return any(self._mesh_intersects(pair[0], pair[1]) for pair in pairs)
    
    @staticmethod
    def _mesh_intersects(obj_a, obj_b, depsgraph=None) -> bool:
        """
        Returns True if the meshes of obj_a and obj_b intersect.
        Works on both mesh and evaluated (modifiers applied) data.
        """
        if depsgraph is None:
            depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get evaluated (modifier‑applied) objects
        eval_a = obj_a.evaluated_get(depsgraph)
        eval_b = obj_b.evaluated_get(depsgraph)
        mesh_a = eval_a.to_mesh()
        mesh_b = eval_b.to_mesh()

        # Create BMeshes and load meshes
        bm_a = bmesh.new()
        bm_b = bmesh.new()
        bm_a.from_mesh(mesh_a)
        bm_b.from_mesh(mesh_b)

        # Transform BMeshes into world space
        bm_a.transform(eval_a.matrix_world)
        bm_b.transform(eval_b.matrix_world)

        # Build BVH trees from the transformed BMeshes
        tree_a = BVHTree.FromBMesh(bm_a)
        tree_b = BVHTree.FromBMesh(bm_b)

        # Cleanup
        eval_a.to_mesh_clear()
        eval_b.to_mesh_clear()
        bm_a.free()
        bm_b.free()

        # Check for any overlapping triangles
        return bool(tree_a.overlap(tree_b))
       
    def step(self, actuator_velocities: list[float] | None = None, grapper_state: bool | None = None) -> float:
        """updates the environment"""
        # set new action params
        self.current_velocites = actuator_velocities if actuator_velocities is not None else self.current_velocites
        next_grapper_state = self.grapper_state if grapper_state is None else bool(grapper_state)
        grapper_moved = next_grapper_state != self.grapper_state
        self.grapper_state = next_grapper_state

        # move actuators
        self.robot_objects["base"].rotation_euler.z -= radians(self.current_velocites[0] / self.fps)
        self.robot_objects["primary arm"].rotation_euler.x += radians(self.current_velocites[1] / self.fps)
        self.robot_objects["secondary arm - part 1"].rotation_euler.x -= radians(self.current_velocites[2] / self.fps)
        self.robot_objects["secondary arm - part 2"].rotation_euler.z -= radians(self.current_velocites[3] / self.fps)
        self.robot_objects["tertiary arm - part 1"].rotation_euler.x += radians(self.current_velocites[4] / self.fps)
        self.robot_objects["tertiary arm - part 2"].rotation_euler.y -= radians(self.current_velocites[5] / self.fps)

        # set grapper

        if self.grapper_state:
            self.robot_objects["finger - left"].rotation_euler.z = radians(160)
            self.robot_objects["finger - right"].rotation_euler.z = radians(190)
        else:
            self.robot_objects["finger - left"].rotation_euler.z = radians(235)
            self.robot_objects["finger - right"].rotation_euler.z = radians(125)
        
        # calculate cost
        max_velocities = [6.7, 6.7, 6.7, 9.5, 6.7, 9.5]
        max_current = [0.7, 1.7, 1.5, 0.7, 0.7, 0.7]
        return sum([abs(v_c) / v_m * a_m for v_c, v_m, a_m in zip(self.current_velocites, max_velocities, max_current)] + [0.1 if grapper_moved else 0])


# server functions
HOST = 'localhost'
PORT = 5055
VERBOSE_REQUEST_LOGS = False

def send_response(connection, response_data):
    """Encodes and sends a response dictionary to the client."""
    try:
        # Serialize the response dictionary to a JSON string, then encode to bytes
        msg_data = json.dumps(response_data).encode('utf-8')
        # Pack the message length into a 4-byte header
        header = struct.pack('>I', len(msg_data))
        # Send the header followed by the message data
        connection.sendall(header + msg_data)
    except Exception as e:
        print(f"Error sending response: {e}")

# Rotation settings
#                                     Fingers: rig, lef
# Home Rotations:   0, 0, 0, 0, 0, 0, Fingers: 180, 180
# Rotation changes: -, +, -, -, +, -, Fingers: +  , -

class RLServerModalOperator(bpy.types.Operator):
    """Run RL Env Server in Blender Modal Timer"""
    bl_idname = "wm.rl_env_server_modal"
    bl_label = "Start RL Env Server (Modal Timer)"

    _timer = None
    _server_socket = None
    _client_conn = None
    _client_addr = None

    def modal(self, context, event):
        if event.type == 'TIMER':
            # Accept new connection if none
            if self._client_conn is None:
                try:
                    self._client_conn, self._client_addr = self._server_socket.accept()
                    self._client_conn.setblocking(False)
                    print(f"Connected by {self._client_addr}")
                except BlockingIOError:
                    pass  # No connection yet

            # Handle client requests
            if self._client_conn:
                try:
                    header = self._client_conn.recv(4)
                    if header:
                        msg_len = struct.unpack('>I', header)[0]
                        data = b''
                        while len(data) < msg_len:
                            packet = self._client_conn.recv(msg_len - len(data))
                            if not packet:
                                break
                            data += packet
                        if data:
                            request = json.loads(data.decode())
                            # Handle request as before
                            if request["function"] == "reset":
                                env.reset(request["args"]["cube_position"], request["args"]["robot_pose"])
                                if VERBOSE_REQUEST_LOGS:
                                    print(f'reset environment to cube_position={request["args"]["cube_position"]}, robot_pose={request["args"]["robot_pose"]}')
                            
                            if request["function"] == "get_state":
                                result = env.get_state(
                                    request["args"]["actuator_rotations"],
                                    request["args"]["actuator_velocities"],
                                    request["args"]["target_cube_state"],
                                    request["args"]["graper"],
                                    request["args"]["collisions"],
                                    request["args"]["workplate_coverage"],
                                    request["args"]["distance_to_target"],
                                    request["args"]["image"],
                                )
                                send_response(self._client_conn, {"result": result})
                                if VERBOSE_REQUEST_LOGS:
                                    print('retrieved and send environment state')
                            
                            if request["function"] == "step":
                                result = env.step(
                                    request["args"]["actuator_velocities"],
                                    request["args"]["grapper_state"]
                                )
                                send_response(self._client_conn, {"result": result})
                                if VERBOSE_REQUEST_LOGS:
                                    print(f'updated env with actuator_velocities={request["args"]["actuator_velocities"]}, grapper_state={request["args"]["grapper_state"]} resulting in a cost of {result}')
                            
                            if request["function"] == "target_cube_in_view":
                                result = env.target_cube_in_view()
                                send_response(self._client_conn, {"result": result})
                                if VERBOSE_REQUEST_LOGS:
                                    print(f'checked if target cube is in view: {result}')

                            if request["function"] == "target_cube_within_padding":
                                result = env.target_cube_within_padding(request["args"].get("padding", 0.1))
                                send_response(self._client_conn, {"result": result})
                                if VERBOSE_REQUEST_LOGS:
                                    print(
                                        "checked padded cube visibility: "
                                        f"padding={request['args'].get('padding', 0.1)} result={result}"
                                    )

                            if request["function"] == "cube_visibility_labels":
                                result = env.cube_visibility_labels()
                                send_response(self._client_conn, {"result": result})
                                if VERBOSE_REQUEST_LOGS:
                                    print(f"computed cube visibility labels: {result}")

                            if request["function"] == "set_robot_pose":
                                env.set_robot_pose(request["args"]["actuator_rotations"])
                                if VERBOSE_REQUEST_LOGS:
                                    print(f'set robot pose to {request["args"]["actuator_rotations"]}')

                            if request["function"] == "set_cube_pose":
                                env.set_cube_pose(
                                    request["args"]["x"],
                                    request["args"]["y"],
                                    request["args"].get("z", 0.025),
                                    request["args"].get("yaw"),
                                )
                                if VERBOSE_REQUEST_LOGS:
                                    print(
                                        "set cube pose to "
                                        f"x={request['args']['x']}, y={request['args']['y']}, "
                                        f"z={request['args'].get('z', 0.025)}, yaw={request['args'].get('yaw')}"
                                    )
                
                except BlockingIOError:
                    pass  # No data yet
                except Exception as e:
                    print(f"Client error: {e}")
                    self._client_conn.close()
                    self._client_conn = None
                    self._client_addr = None

        return {'PASS_THROUGH'}

    def execute(self, context):
        # Setup non-blocking server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((HOST, PORT))
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)
        print("RL Env Server started (modal timer)")

        wm = context.window_manager
        tick_rate_multiplier = 3.0
        effective_tick_hz = max(getattr(env, "fps", 30), 1) * tick_rate_multiplier
        timer_interval_s = 1.0 / effective_tick_hz
        self._timer = wm.event_timer_add(timer_interval_s, window=context.window)
        print(
            "RL Env Server timer interval set to "
            f"{timer_interval_s:.4f}s (effective {effective_tick_hz:.1f} Hz)"
        )
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        if self._client_conn:
            self._client_conn.close()
        if self._server_socket:
            self._server_socket.close()
        print("RL Env Server stopped.")

def register():
    bpy.utils.register_class(RLServerModalOperator)

def unregister():
    bpy.utils.unregister_class(RLServerModalOperator)

if __name__ == "__main__":
    env = RobotEnv(fps=30)
    register()
    # To start: F3 > "Start RL Env Server (Modal Timer)"
    # Run script and then run in the `bpy.ops.wm.rl_env_server_modal()` console
