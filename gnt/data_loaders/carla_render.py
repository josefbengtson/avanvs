import os
import numpy as np
import imageio
import torch
import sys
import random

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import get_nearest_pose_ids
from .carla_data_utils import load_carla_data, load_carla_data_two_conditions
from .carla_data_utils import batch_parse_llff_poses, batch_parse_llff_poses_inv


class CarlaRenderDataset(Dataset):
    def __init__(self, args, scenes="fern", **kwargs):
        render_source_folder = args.render_folder[0]
        self.folder_path = os.path.join(args.rootdir, "data/{}/".format(render_source_folder))
        self.num_source_views = args.num_source_views


        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.h = []
        self.w = []
        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.train_rgb_files_cond2 = []

        # Loading conditions as text and converting to list of integers
        conditions_text = args.conditions[0]
        if conditions_text == "Default":
            conditions = []
        elif len(conditions_text) == 3:
            conditions = [int(conditions_text[1])]
        elif len(conditions_text) == 5:
            conditions = [int(conditions_text[1]), int(conditions_text[3])]
        elif len(conditions_text) == 7:
            conditions = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5])]
        elif len(conditions_text) == 9:
            conditions = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5]), int(conditions_text[7])]
        else:
            sys.exit("FAULTY CONDITIONS")

        print("CONDITIONS: ", conditions)
        self.num_conditions = len(conditions)
        if len(scenes) == 0:
            scenes = [name for name in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, name))]
        print("loading {} for rendering".format(scenes))
        print("Number loaded scenes: ", len(scenes))

        self.scene_names = scenes
        print("self.scene_names: ", self.scene_names)
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            print("self.folder_path: ", self.folder_path)
            print("scene_path: ", scene_path)
            if len(conditions) == 0:
                _, poses, bds, render_poses, i_test, rgb_files_cond1 = load_carla_data(
                    scene_path, load_imgs=False, factor=4, num_views=args.N_views)
                conditions = []
            elif len(conditions) == 1:
                _, poses, bds, render_poses, i_test, rgb_files_cond1, rgb_files_cond2 = load_carla_data_two_conditions(
                    scene_path, conditions, load_imgs=False, factor=4, num_views=args.N_views)
            elif len(conditions) == 2:
                _, poses, bds, render_poses, i_test, rgb_files_cond1, rgb_files_cond2 = load_carla_data_two_conditions(
                    scene_path, conditions, load_imgs=False, factor=4, num_views=args.N_views)
            else:
                print("More than 2 condtions!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break

            near_depth = np.min(bds)
            far_depth = np.max(bds)

            # intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            intrinsics, c2w_mats = batch_parse_llff_poses_inv(poses)

            h, w = poses[0][:2, -1]

            render_intrinsics, render_c2w_mats = batch_parse_llff_poses_inv(render_poses)
            i_test = [i_test]
            i_val = i_test
            i_train = np.array(
                [i for i in np.arange(len(rgb_files_cond1)) if (i not in i_test and i not in i_val)]
            )

            f = 400
            w = 800
            h = 600
            K = np.array([[f, 0, w/2, 0], [0, f, h/2, 0],
                         [0, 0, 1, 0], [0, 0, 0, 1]])

            for idx in range(len(intrinsics)):
                intrinsics[idx, :, :] = K

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files_cond1)[i_train].tolist())
            if self.num_conditions == 2:
                self.train_rgb_files_cond2.append(np.array(rgb_files_cond2)[i_train].tolist())
            self.conditions = conditions

            num_render = len(render_intrinsics)
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])
            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)
            self.h.extend([int(h)] * num_render)
            self.w.extend([int(w)] * num_render)


    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]
        train_set_id = self.render_train_set_ids[idx]
        scene_name = self.scene_names[train_set_id]

        train_rgb_files = self.train_rgb_files[train_set_id]
        if self.num_conditions == 2:
            train_rgb_files_cond2 = self.train_rgb_files_cond2[train_set_id]

        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        id_render = -1

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            self.num_source_views,
            tar_id=id_render,
            angular_dist_method="dist",
        )
        src_rgbs = []
        src_rgbs_cond2 = []

        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)
            if self.num_conditions == 2:
                src_rgb_cond2 = imageio.imread(train_rgb_files_cond2[id]).astype(np.float32) / 255.0
                src_rgbs_cond2.append(src_rgb_cond2)

        src_rgbs = np.stack(src_rgbs, axis=0)
        if self.num_conditions == 2:
            src_rgbs_cond2 = np.stack(src_rgbs_cond2, axis=0)
            src_rgbs_cond2_out = torch.from_numpy(src_rgbs_cond2[..., :3])


        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])
        if self.num_conditions == 2:
            return {
                "camera": torch.from_numpy(camera),
                "rgb_path": "",
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
                "src_rgbs_cond2": src_rgbs_cond2_out,
                "src_cameras": torch.from_numpy(src_cameras),
                "depth_range": depth_range,
                "num_conditions": 2,
                "conditions": self.conditions,
                "scene_name": scene_name
            }
        elif self.num_conditions == 1:
            return {
                "camera": torch.from_numpy(camera),
                "rgb_path": "",
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
                "src_cameras": torch.from_numpy(src_cameras),
                "depth_range": depth_range,
                "num_conditions": 1,
                "conditions": self.conditions,
                "scene_name": scene_name
            }
        elif self.num_conditions == 0:
            return {
                "camera": torch.from_numpy(camera),
                "rgb_path": "",
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
                "src_cameras": torch.from_numpy(src_cameras),
                "depth_range": depth_range,
                "num_conditions": 1,
                "conditions": self.conditions,
                "scene_name": scene_name
            }
        else:
            print("Number of conditions not 1 or 2!!!!!!!!!!!!!!!!!")
