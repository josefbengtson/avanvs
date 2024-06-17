import sys

sys.path.append("../")
import os
import numpy as np
from PIL import Image
import imageio
import torch
from torch.utils.data import Dataset
from .data_utils import quaternion_about_axis, quaternion_matrix, random_crop, random_flip
from .data_utils import get_nearest_pose_ids
import math
import json


def view_obj2camera_rgb(view):
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    h_in_view, w_in_view = view.shape
    rgb = imageio.imread(image_path).astype(np.float32) / 255.0
    h_img, w_img = rgb.shape[:2]
    if h_in_view != h_img or w_in_view != w_img:
        intrinsics[0] *= w_img / w_in_view
        intrinsics[1] *= h_img / h_in_view
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    ref_camera = np.concatenate([list(rgb.shape[:2]), intrinsics_4x4.flatten(), c2w.flatten()])
    return ref_camera, rgb

def generate_render_camera(view,c2w):
    intrinsics = view.camera.intrinsics
    h_in_view, w_in_view = view.shape
    # if h_in_view != h_img or w_in_view != w_img:
    #     intrinsics[0] *= w_img / w_in_view
    #     intrinsics[1] *= h_img / h_in_view
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w_from_view = view.camera.w_f_c


    # c2w = c2w
    render_camera = np.concatenate([list(np.array([h_in_view, w_in_view])), intrinsics_4x4.flatten(), c2w.flatten()])
    return render_camera

def view_obj2camera_rgb_path(view):
    img_size = view.shape
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    return image_path, img_size, intrinsics_4x4, c2w


def sample_target_view_for_training(views, input_rig_id, input_ids):
    input_rig_views = views[input_rig_id]
    input_cam_positions = np.array([input_rig_views[i].camera.w_f_c[:3, 3] for i in input_ids])

    remaining_rig_ids = []
    remaining_cam_ids = []

    for i, rig in enumerate(views):
        for j, cam in enumerate(rig):
            if i == input_rig_id and j in input_ids:
                continue
            else:
                cam_loc = views[i][j].camera.w_f_c[:3, 3]
                # if i != input_rig_id:
                #     print(np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)))
                if np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)) < 0.15:
                    remaining_rig_ids.append(i)
                    remaining_cam_ids.append(j)

    selected_id = np.random.choice(len(remaining_rig_ids))
    selected_view = views[remaining_rig_ids[selected_id]][remaining_cam_ids[selected_id]]
    return selected_view


def get_all_views_in_scene(all_views):
    cameras = []
    rgbs = []
    for rig in all_views:
        for i in range(len(rig)):
            camera, rgb = view_obj2camera_rgb(rig[i])
            cameras.append(camera)
            rgbs.append(rgb)
    return cameras, rgbs


def get_all_views_in_scene_cam_path(all_views):
    c2w_mats = []
    intrinsicss = []
    rgb_paths = []
    img_sizes = []
    for rig in all_views:
        for i in range(len(rig)):
            image_path, img_size, intrinsics_4x4, c2w = view_obj2camera_rgb_path(rig[i])
            rgb_paths.append(image_path)
            intrinsicss.append(intrinsics_4x4)
            c2w_mats.append(c2w)
            img_sizes.append(img_size)
    return rgb_paths, img_sizes, intrinsicss, c2w_mats

def get_all_views_in_scene_cam_path_one_rig(all_views):
    c2w_mats = []
    intrinsicss = []
    rgb_paths = []
    img_sizes = []
    for rig in all_views:
        # for i in range(len(rig)):
        i = 0
        image_path, img_size, intrinsics_4x4, c2w = view_obj2camera_rgb_path(rig[i])
        rgb_paths.append(image_path)
        intrinsicss.append(intrinsics_4x4)
        c2w_mats.append(c2w)
        img_sizes.append(img_size)
    return rgb_paths, img_sizes, intrinsicss, c2w_mats

def sort_nearby_views_by_angle(query_pose, ref_poses):
    query_direction = np.sum(query_pose[:3, 2:4], axis=-1)
    query_direction = query_direction / np.linalg.norm(query_direction)
    ref_directions = np.sum(ref_poses[:, :3, 2:4], axis=-1)
    ref_directions = ref_directions / np.linalg.norm(ref_directions, axis=-1, keepdims=True)
    inner_product = np.sum(ref_directions * query_direction[None, ...], axis=1)
    sorted_inds = np.argsort(inner_product)[::-1]
    return sorted_inds


class Camera(object):
    """Represents a Camera with intrinsics and world from/to camera transforms.
    Attributes:
      w_f_c: The world from camera 4x4 matrix.
      c_f_w: The camera from world 4x4 matrix.
      intrinsics: The camera intrinsics as a 3x3 matrix.
      inv_intrinsics: The inverse of camera intrinsics matrix.
    """

    def __init__(self, intrinsics, w_f_c):
        """Constructor.
        Args:
          intrinsics: A numpy 3x3 array representing intrinsics.
          w_f_c: A numpy 4x4 array representing wFc.
        """
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        self.w_f_c = w_f_c
        self.c_f_w = np.linalg.inv(w_f_c)


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


def _WorldFromCameraFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.
    Args:
        view_json: A dictionary of view parameters.
    Returns:
        A 4x4 transform matrix representing the world from camera transform.
    """
    transform = np.identity(4)
    position = view_json["position"]
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json["orientation"]
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform

    axis = angle_axis / angle
    rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return transform


def _IntrinsicsFromViewDict(view_params):
    """Fills the intrinsics matrix from view_params.
    Args:
        view_params: Dict view parameters.
    Returns:
        A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = np.identity(3)
    intrinsics[0, 0] = view_params["focal_length"]
    intrinsics[1, 1] = view_params["focal_length"] * view_params["pixel_aspect_ratio"]
    intrinsics[0, 2] = view_params["principal_point"][0]
    intrinsics[1, 2] = view_params["principal_point"][1]
    return intrinsics


def ReadView(base_dir, view_json):
    return View(
        image_path=os.path.join(base_dir, view_json["relative_path"]),
        shape=(int(view_json["height"]), int(view_json["width"])),
        camera=Camera(_IntrinsicsFromViewDict(view_json), _WorldFromCameraFromViewDict(view_json)),
    )


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, "models.json")) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views


def InterpolateDepths(near_depth, far_depth, num_depths):
    """Returns num_depths from (far_depth, near_depth), interpolated in inv depth.
    Args:
        near_depth: The first depth.
        far_depth: The last depth.
        num_depths: The total number of depths to create, include near_depth and
        far_depth are always included and other depths are interpolated between
        them, in inverse depth space.
    Returns:
        The depths sorted in descending order (so furthest first). This order is
        useful for back to front compositing.
    """

    inv_near_depth = 1.0 / near_depth
    inv_far_depth = 1.0 / far_depth
    depths = []
    for i in range(0, num_depths):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_far_depth + (inv_near_depth - inv_far_depth) * fraction
        depths.append(1.0 / inv_depth)
    return depths


def ReadViewImages(views):
    """Reads the images for the passed views."""
    for view in views:
        # Keep images unnormalized as uint8 to save RAM and transmission time to
        # and from the GPU.
        view.image = np.array(Image.open(view.image_path))


def WriteNpToImage(np_image, path):
    """Writes an image as a numpy array to the passed path.
        If the input has more than four channels only the first four will be
        written. If the input has a single channel it will be duplicated and
        written as a three channel image.
    Args:
        np_image: A numpy array.
        path: The path to write to.
    Raises:
        IOError: if the image format isn't recognized.
    """

    min_value = np.amin(np_image)
    max_value = np.amax(np_image)
    if min_value < 0.0 or max_value > 255.1:
        print("Warning: Outside image bounds, min: %f, max:%f, clipping.", min_value, max_value)
        np.clip(np_image, 0.0, 255.0)
    if np_image.shape[2] == 1:
        np_image = np.concatenate((np_image, np_image, np_image), axis=2)

    if np_image.shape[2] == 3:
        image = Image.fromarray(np_image.astype(np.uint8))
    elif np_image.shape[2] == 4:
        image = Image.fromarray(np_image.astype(np.uint8), "RGBA")

    _, ext = os.path.splitext(path)
    ext = ext[1:]
    if ext.lower() == "png":
        image.save(path, format="PNG")
    elif ext.lower() in ("jpg", "jpeg"):
        image.save(path, format="JPEG")
    else:
        raise IOError("Unrecognized format for %s" % path)


class SpacesFreeDatasetRender(Dataset):
    def __init__(self, args, scenes, **kwargs):
        print("Creating Spaces Dataset")


        # Load Conditions

        conditions_text = args.conditions[0]
        # Load conditions as text and convert to a list
        if conditions_text == "Default":
            conditions = []
        elif len(conditions_text) == 3:
            conditions = [int(conditions_text[1])]
        elif len(conditions_text) == 5:
            conditions = [int(conditions_text[1]), int(conditions_text[3])]
        elif len(conditions_text) == 7:
            conditions = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5])]
        elif len(conditions_text) == 9:
            conditions = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5]),
                          int(conditions_text[7])]
        else:
            sys.exit("FAULTY CONDITIONS")

        print("Conditions: ", conditions)
        print("num conditions: ", len(conditions))
        self.num_conditions = len(conditions)
        self.conditions = conditions


        self.folder_path = os.path.join(args.rootdir, "data/spaces_dataset/data/800/")
        self.num_source_views = args.num_source_views

        # eval_scene_ids = [0, 9, 10, 23, 24, 52, 56, 62, 63, 73]

        eval_scene_ids = [int(scenes[0])]
        # eval_scene_ids = [46,48]
        self.scene_dirs = [
            os.path.join(self.folder_path, "scene_{:03d}".format(i)) for i in eval_scene_ids
        ]
        print("self.scene_dirs: ", self.scene_dirs)
        self.all_views_scenes = []
        self.all_rgb_paths_scenes = []
        self.all_intrinsics_scenes = []
        self.all_img_sizes_scenes = []
        self.all_c2w_scenes = []
        self.all_render_poses = []
        self.all_render_intrinsics = []

        for scene_dir in self.scene_dirs:
            views = ReadScene(scene_dir)
            self.all_views_scenes.append(views)
            rgb_paths, img_sizes, intrinsics, c2w_mats_one = get_all_views_in_scene_cam_path_one_rig(views)
            rgb_paths, img_sizes, intrinsicss, c2w_mats = get_all_views_in_scene_cam_path(views)

            self.all_rgb_paths_scenes.append(rgb_paths)
            self.all_img_sizes_scenes.append(img_sizes)
            self.all_intrinsics_scenes.append(intrinsicss)
            self.all_c2w_scenes.append(c2w_mats)

            # Generate poses for spiral path
            num_c2w_mats_one = len(c2w_mats_one)
            num_c2w_mats = len(c2w_mats)

            c2w_mats_shape = c2w_mats[0].shape
            c2w_mats = np.zeros([num_c2w_mats_one,c2w_mats_shape[0], c2w_mats_shape[1]])
            for i in range(num_c2w_mats_one):
                c2w_mats[i, :, :] = c2w_mats_one[i]

            c2w = poses_avg(c2w_mats)

            ## Get spiral
            # Get average pose
            up = normalize(c2w_mats[:, :3, 1].sum(0))

            # Find a reasonable "focus depth" for this dataset
            near_depth = 0.7
            far_depth = 100

            close_depth, inf_depth = near_depth * 0.9, far_depth * 5.0
            dt = 0.75
            mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
            focal = mean_dz

            # Get radii for spiral path
            shrink_factor = 0.8
            zdelta = close_depth * 0.2
            tt = c2w_mats[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = c2w
            N_views = args.N_views
            render_poses = render_path_spiral(
                c2w_path, up, rads, focal, zdelta, zrate=0.5, N=N_views)
            intrinsics, c2w_mats = batch_parse_llff_poses(render_poses)

            self.all_render_poses.append(c2w_mats)
            self.all_render_intrinsics.append(intrinsics)


    def __len__(self):
        # return len(self.all_views_scenes)
        return len(self.all_render_poses[0])

    def __getitem__(self, render_id):
        idx = 0
        all_views = self.all_views_scenes[idx]

        # Extracting one view to get camera parameters
        rig_selected = all_views[0]
        cam_selected = rig_selected[0]

        intrinsics_3x3 = cam_selected.camera.intrinsics
        h, w = cam_selected.shape
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = intrinsics_3x3

        render_pose = self.all_render_poses[idx][render_id]

        render_camera = np.concatenate(([h, w], intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )
        # render_camera = render_pose
        all_c2w_mats = self.all_c2w_scenes[idx]
        all_rgb_paths = self.all_rgb_paths_scenes[idx]
        all_intrinsics = self.all_intrinsics_scenes[idx]
        all_img_sizes = self.all_img_sizes_scenes[idx]

        id_render = -1
        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            np.array(all_c2w_mats),
            self.num_source_views,
            tar_id=id_render,
            angular_dist_method="dist",
        )

        ref_cameras = []
        ref_rgbs = []
        w_max, h_max = 0, 0

        for id in nearest_pose_ids:
            rgb_path = all_rgb_paths[id]
            ref_rgb = imageio.imread(rgb_path).astype(np.float32) / 255.0
            h_in_view, w_in_view = all_img_sizes[id]
            h_img, w_img = ref_rgb.shape[:2]
            ref_rgbs.append(ref_rgb)
            ref_intrinsics = all_intrinsics[id]
            if h_in_view != h_img or w_in_view != w_img:
                ref_intrinsics[0] *= w_img / w_in_view
                ref_intrinsics[1] *= h_img / h_in_view
            ref_c2w = all_c2w_mats[id]
            ref_camera = np.concatenate(
                [list(ref_rgb.shape[:2]), ref_intrinsics.flatten(), ref_c2w.flatten()]
            )
            ref_cameras.append(ref_camera)
            h, w = ref_rgb.shape[:2]
            w_max = max(w, w_max)
            h_max = max(h, h_max)

        ref_rgbs_np = np.ones((len(ref_rgbs), h_max, w_max, 3), dtype=np.float32)
        for i, ref_rgb in enumerate(ref_rgbs):
            orig_h, orig_w = ref_rgb.shape[:2]
            h_start = int((h_max - orig_h) / 2.0)
            w_start = int((w_max - orig_w) / 2.0)
            ref_rgbs_np[i, h_start : h_start + orig_h, w_start : w_start + orig_w] = ref_rgb
            ref_cameras[i][4] += (w_max - orig_w) / 2.0
            ref_cameras[i][8] += (h_max - orig_h) / 2.0
            ref_cameras[i][0] = h_max
            ref_cameras[i][1] = w_max

        ref_cameras = np.array(ref_cameras)
        near_depth = 0.7
        far_depth = 100
        depth_range = torch.tensor([near_depth, far_depth])
        return {
            "camera": torch.from_numpy(render_camera).float(),
            "rgb_path": "",
            "src_rgbs": torch.from_numpy(ref_rgbs_np).float(),
            "src_cameras": torch.from_numpy(np.stack(ref_cameras, axis=0)).float(),
            "depth_range": depth_range,
            "num_conditions": self.num_conditions,
            "conditions": self.conditions
        }

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    # theta_start = 1.5 * np.pi
    # theta_end = 1.5 * np.pi
    theta_start = 0.5 * np.pi
    theta_end = 2.5 * np.pi

    dz_start = 0
    num_poses_per_theta = int(N)
    zrate = 1

    add_render_poses_theta(theta_start, theta_end, dz_start, num_poses_per_theta, render_poses, c2w, rads, focal, up, hwf, zrate)

    return render_poses


def add_render_poses_theta(theta_start, theta_end, dz, num_poses, render_poses, c2w, rads, focal, up, hwf, zrate):
    for theta in np.linspace(theta_start, theta_end, num_poses + 1):

        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        new_pose = np.concatenate([viewmatrix(z, up, c), hwf], 1)
        new_pose[2, 3] = new_pose[2, 3] + dz
        render_poses.append(new_pose)

def add_render_poses_z(dz_start, dz_end, theta, num_poses, render_poses, c2w, rads, focal, up, hwf):
    for dz in np.linspace(dz_start, dz_end, num_poses + 1):
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))

        new_pose = np.concatenate([viewmatrix(z, up, c), hwf], 1)
        new_pose[2, 3] = new_pose[2, 3] + dz
        render_poses.append(new_pose)

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def batch_parse_llff_poses(poses):
    all_intrinsics = []
    all_c2w_mats = []
    for pose in poses:
        intrinsics, c2w_mat = parse_llff_pose(pose)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats

def parse_llff_pose(pose):
    """
    convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    """
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    c2w_4x4[:, 1:3] *= 1 # Originally -1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    intrinsics = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return intrinsics, c2w_4x4


