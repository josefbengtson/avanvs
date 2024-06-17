import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .carla_data_utils import load_carla_data, batch_parse_llff_poses, batch_parse_llff_poses_inv
from .carla_data_utils import load_carla_data_two_conditions

from scipy.io import savemat
import random

class CarlaTestDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join(args.rootdir, "data/carla_test/")
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_poses = []

        # Intialize for unseen poses
        if args.regularize_unseen:
            self.unseen_poses = []
        else:
            self.unseen_poses = None
        self.render_intrinsics = []
        self.render_rgb_files = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.train_rgb_files_cond2 = []

        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------
        # Choose if change appearance
        self.appearance_change = True
        # ------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        if mode == "train" and len(scenes) != 1:
            if "TEST_Scene122" in scenes:
                print("Removing Test Scene from Training")
                scenes.remove("TEST_Scene122")
        # Extract conditions

        conditions_text = args.conditions[0]
        # print("conditions text: ", conditions_text)
        if conditions_text == "Default":
            conditions_load = []
        elif len(conditions_text) == 3:
            conditions_load = [int(conditions_text[1])]
            # num_conditions = 1
        elif len(conditions_text) == 5:
            conditions_load = [int(conditions_text[1]), int(conditions_text[3])]
        elif len(conditions_text) == 7:
            conditions_load = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5])]
        elif len(conditions_text) == 9:
            conditions_load = [int(conditions_text[1]), int(conditions_text[3]), int(conditions_text[5]), int(conditions_text[7])]
        else:
            sys.exit("FAULTY CONDITIONS")
            return

        num_conditions = len(conditions_load)
        if num_conditions == 4:
            num_scenes_factor = 1
        else:
            num_scenes_factor = 1
        conditions = np.load(args.rootdir + "/data/conditions_list.npy")

        self.num_conditions = len(conditions_load)

        if self.num_conditions > 1:
            self.render_rgb_files_cond2 = []

        if self.appearance_change or self.num_conditions == 0:
            for j in range(num_scenes_factor):
                for i, scene in enumerate(scenes):
                    scene_path = os.path.join(self.folder_path, scene)
                    if mode == "train":
                        scene_conditions = conditions[i,:]
                    else:
                        scene_conditions = np.random.choice(conditions_load, size=2, replace=False)
                        conditions = np.zeros([len(scenes), 2])
                        conditions[:, 0] = scene_conditions[0]
                        conditions[:, 1] = scene_conditions[1]

                    if len(scene_conditions) == 0:
                        _, poses, bds, render_poses, i_test, rgb_files_cond1 = load_carla_data(
                            scene_path, load_imgs=False, factor=4, num_views=args.N_views)
                    elif len(scene_conditions) == 1:
                        _, poses, bds, render_poses, i_test, rgb_files_cond1, rgb_files_cond2 = load_carla_data_two_conditions(
                            scene_path, scene_conditions, load_imgs=False, factor=4, num_views=args.N_views)
                    elif len(scene_conditions) == 2:
                        _, poses, bds, render_poses, i_test, rgb_files_cond1, rgb_files_cond2 = load_carla_data_two_conditions(
                            scene_path, scene_conditions, load_imgs=False, factor=4, num_views=args.N_views)
                    else:
                        print("More than 2 condtions!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        break

                    near_depth = np.min(bds)
                    far_depth = np.max(bds)
                    intrinsics, c2w_mats = batch_parse_llff_poses_inv(poses)
                    render_intrinsics, render_c2w_mats = batch_parse_llff_poses_inv(render_poses)
                    i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
                    i_train = np.array(
                        [
                            j
                            for j in np.arange(int(poses.shape[0]))
                            if (j not in i_test and j not in i_test)
                        ]
                    )

                    if mode == "train":
                        i_render = i_train
                    else:
                        i_render = i_test

                    self.train_intrinsics.append(intrinsics[i_train])
                    self.train_poses.append(c2w_mats[i_train])
                    self.train_rgb_files.append(np.array(rgb_files_cond1)[i_train].tolist())

                    if self.num_conditions > 1:
                        if len(np.array(rgb_files_cond2)) < 10:
                            print("Scene short: ", scene)
                        self.train_rgb_files_cond2.append(np.array(rgb_files_cond2)[i_train].tolist())
                        self.render_rgb_files_cond2.extend(np.array(rgb_files_cond2)[i_render].tolist())

                    num_render = len(i_render)
                    self.render_rgb_files.extend(np.array(rgb_files_cond1)[i_render].tolist())


                    self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
                    self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
                    # Added unseen pose list
                    if args.regularize_unseen:
                        self.unseen_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])


                    self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
                    self.render_train_set_ids.extend([i] * num_render)
        else:
            print("Hello in second section")
            sys.exit("Entering unused section of carla_test")
            for i, scene in enumerate(scenes):
                print("scene: ", scene)
                for j in range(self.num_conditions):
                    print("j: ", j)
                    scene_path = os.path.join(self.folder_path, scene)
                    condition = [conditions[j]]

                    _, poses, bds, render_poses, i_test, rgb_files_cond1, rgb_files_cond2 = load_carla_data_two_conditions(
                        scene_path, condition, load_imgs=False, factor=4, num_views=args.N_views)


                    near_depth = np.min(bds)
                    far_depth = np.max(bds)
                    # Load inverted poses to get c2w
                    intrinsics, c2w_mats = batch_parse_llff_poses_inv(poses)
                    render_intrinsics, render_c2w_mats = batch_parse_llff_poses_inv(render_poses)

                    i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
                    i_train = np.array(
                        [
                            j
                            for j in np.arange(int(poses.shape[0]))
                            if (j not in i_test and j not in i_test)
                        ]
                    )

                    if mode == "train":
                        i_render = i_train
                    else:
                        i_render = i_test

                    self.train_intrinsics.append(intrinsics[i_train])
                    self.train_poses.append(c2w_mats[i_train])
                    self.train_rgb_files.append(np.array(rgb_files_cond1)[i_train].tolist())

                    num_render = len(i_render)
                    self.render_rgb_files.extend(np.array(rgb_files_cond1)[i_render].tolist())

                    self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
                    self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
                    # Added unseen pose list
                    if args.regularize_unseen:
                        self.unseen_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])

                    self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
                    self.render_train_set_ids.extend([i] * num_render)
        self.conditions = conditions

    def __len__(self):
        return (
            len(self.render_rgb_files)
        if self.mode == "train"
            else len(self.render_rgb_files)
        )

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0

        if self.num_conditions > 1 and self.appearance_change:
            rgb_file_cond2 = self.render_rgb_files_cond2[idx]
            rgb_cond2 = imageio.imread(rgb_file_cond2).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]

        # Extract one of the unseen poses randomly
        if self.unseen_poses is not None:
            num_unseen_poses = len(self.unseen_poses)
            unseen_index = np.random.randint(0, num_unseen_poses)
            unseen_pose = self.unseen_poses[unseen_index]

        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        if self.num_conditions > 1 and self.appearance_change:
            train_rgb_files_cond2 = self.train_rgb_files_cond2[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)
        if self.unseen_poses is not None:
            camera_unseen = np.concatenate(
                (list(img_size), intrinsics.flatten(), unseen_pose.flatten())
            ).astype(np.float32)
        else:
            camera_unseen = None

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 28),
            tar_id=id_render,
            angular_dist_method="dist",
        )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )


        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

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
            if self.num_conditions > 1 and self.appearance_change:
                src_rgb_cond2 = imageio.imread(train_rgb_files_cond2[id]).astype(np.float32) / 255.0
                src_rgbs_cond2.append(src_rgb_cond2)


        src_rgbs = np.stack(src_rgbs, axis=0)



        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == "train" and self.random_crop:
            crop_h = np.random.randint(low=250, high=550)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(600 * (250 / crop_h))
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w

            if (self.num_conditions > 1) and self.appearance_change:
                src_rgbs_cond2 = np.stack(src_rgbs_cond2, axis=0)
                rgb, camera, src_rgbs, src_cameras, rgb_cond2, src_rgbs_cond2 = random_crop(
                    rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w), rgb_cond2=rgb_cond2, src_rgbs_cond2=src_rgbs_cond2
                )
            else:
                rgb, camera, src_rgbs, src_cameras = random_crop(
                    rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
                )
        else:
            if self.num_conditions > 1 and self.appearance_change:
                src_rgbs_cond2 = np.stack(src_rgbs_cond2, axis=0)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])
        if self.num_conditions > 1 and self.appearance_change:

            return {
                "rgb": torch.from_numpy(rgb[..., :3]),
                "rgb_cond2": torch.from_numpy(rgb_cond2[..., :3]),
                "camera": torch.from_numpy(camera),
                "camera_unseen": torch.from_numpy(camera_unseen) if camera_unseen is not None else [],
                "rgb_path": rgb_file,
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
                "src_rgbs_cond2": torch.from_numpy(src_rgbs_cond2[..., :3]),
                "src_cameras": torch.from_numpy(src_cameras),
                "depth_range": depth_range,
                "num_conditions": 2,
                "conditions": self.conditions[train_set_id, :],
                "train_rgb_files": train_rgb_files,
                "train_rgb_files_cond2": train_rgb_files_cond2
            }
        else:
            return {
                "rgb": torch.from_numpy(rgb[..., :3]),
                "camera": torch.from_numpy(camera),
                "camera_unseen": torch.from_numpy(camera_unseen) if camera_unseen is not None else [],
                "rgb_path": rgb_file,
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
                "src_cameras": torch.from_numpy(src_cameras),
                "depth_range": depth_range,
                "conditions": self.conditions[train_set_id, :]
            }

