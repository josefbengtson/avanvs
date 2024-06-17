import numpy as np
import torch
import torch.nn.functional as F
import sys

rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2

    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, patch_size=5, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data["rgb"] if "rgb" in data.keys() else None
        self.rgb_cond2 = data["rgb_cond2"] if "rgb_cond2" in data.keys() else None
        # print("data keys: ", data.keys())
        # print("conditions: ", data["conditions"])
        # sys.exit("In Ray Sampler")
        if "conditions" in data.keys():
            self.conditions = data["conditions"]
        else:
            self.conditions = None
        # self.conditions = [torch.tensor([0])]
        # print("data[conditions]: ", data["conditions"])
        self.H_patch = patch_size
        self.W_patch = patch_size

        if "train_rgb_files" in data.keys():
            self.train_rgb_files = data["train_rgb_files"]
        else:
            self.train_rgb_files = None

        if "train_rgb_files_cond2" in data.keys():
            self.train_rgb_files_cond2 = data["train_rgb_files_cond2"]
        else:
            self.train_rgb_files_cond2 = None


        self.camera = data["camera"]
        # print("camera unseen1: ", len(data["camera_unseen"]))

        if "camera_unseen" in data.keys():
            if len(data["camera_unseen"]) != 0:
                self.camera_unseen = data["camera_unseen"]
            else:
                self.camera_unseen = None
        else:
            self.camera_unseen = None
        self.rgb_path = data["rgb_path"]
        self.depth_range = data["depth_range"]
        self.device = device

        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        if self.camera_unseen is not None:
            W_unseen, H_unseen, self.intrinsics, self.c2w_mat_unseen = parse_camera(self.camera_unseen)

        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(
                    self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor
                ).permute(0, 2, 3, 1)
            if self.rgb_cond2 is not None:
                self.rgb_cond2 = F.interpolate(
                    self.rgb_cond2(0, 3, 1, 2), scale_factor=resize_factor
                ).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_single_image(
            self.H, self.W, self.intrinsics, self.c2w_mat
        )
        if self.camera_unseen is not None:
            self.rays_o_unseen, self.rays_d_unseen = self.get_rays_single_patch(
                self.H, self.W, self.intrinsics, self.c2w_mat_unseen)
        else:
            self.rays_o_unseen = None
            self.rays_d_unseen = None

        # print("self.rays_o shape: ", self.rays_o.shape)
        # print("self.rays_o_unseen shape: ", self.rays_o_unseen.shape)

        # print("self.rays_d shape: ", self.rays_d.shape)
        # print("self.rays_d_unseen shape: ", self.rays_d_unseen.shape)

        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if self.rgb_cond2 is not None:
            self.rgb_cond2 = self.rgb_cond2.reshape(-1, 3)

        if "src_rgbs" in data.keys():
            self.src_rgbs = data["src_rgbs"]
        else:
            self.src_rgbs = None
        # print("data.keys(): ", data.keys())
        if "src_rgbs_cond2" in data.keys():
            # print("Adding cond2")
            self.src_rgbs_cond2 = data["src_rgbs_cond2"]
        else:
            self.src_rgbs_cond2 = None

        if "src_cameras" in data.keys():
            self.src_cameras = data["src_cameras"]
        else:
            self.src_cameras = None

        # print("Hello in RaySAMPLER!")
        # if "num_conditions" in data.keys():
        #     print("Number conditions: ", data["num_conditions"])
        # print("src rgbs shape: ", self.src_rgbs.shape)
        # if "src_rgbs_cond2" in data.keys():
        #     print("src rgbs cond2 shape: ", self.src_rgbs_cond2.shape)


    def get_rays_single_image(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        # print("Render stride: ", self.render_stride)
        u, v = np.meshgrid(
            np.arange(W)[:: self.render_stride], np.arange(H)[:: self.render_stride]
        )
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x HW x 3
        return rays_o, rays_d

    def get_rays_single_patch(self, H, W, intrinsics, c2w):
        """
        Returns the origin and direction of the rays for a single image.

        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return: rays_o: array of shape (batch_size, num_rays, 3) containing the origin of the rays
                 rays_d: array of shape (batch_size, num_rays, 3) containing the direction of the rays
        """
        # Generate a grid of pixel coordinates for the image

        h_patch_lr_corner = np.random.randint(0, H-self.H_patch-1)
        w_patch_lr_corner = np.random.randint(0, W-self.W_patch-1)
        # print("lr corner: (", h_patch_lr_corner,", ", w_patch_lr_corner, ")")
        u, v = np.meshgrid(
            np.arange(W)[:: 1], np.arange(H)[:: 1]
        )
        # print("u shape: ", u.shape, "v shape: ", v.shape)
        # print("u shape: ", u.shape, "v shape: ", v.shape)
        u_patch = u[h_patch_lr_corner:h_patch_lr_corner + self.H_patch, w_patch_lr_corner:w_patch_lr_corner + self.W_patch]
        v_patch = v[h_patch_lr_corner:h_patch_lr_corner + self.H_patch, w_patch_lr_corner:w_patch_lr_corner + self.W_patch]

        # print("u: ", u_patch, "\n v: ", v_patch)


        # Flatten the arrays of pixel coordinates
        u = u_patch.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v_patch.reshape(-1).astype(dtype=np.float32)  # + 0.5
        # Stack the pixel coordinates into a (3, num_rays) array
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, num_rays)
        # Convert the array to a PyTorch tensor
        pixels = torch.from_numpy(pixels)
        # Reshape the tensor to (batch_size, 3, num_rays) and repeat it across the batch dimension
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        # Calculate the direction of the rays using the intrinsic and extrinsic matrices
        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)
        # Reshape the direction tensor to (batch_size, num_rays, 3)
        rays_d = rays_d.reshape(-1, 3)
        # Calculate the origin of the rays
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x num_rays x 3
        return rays_o, rays_d


    def get_all(self):
        if self.camera_unseen is not None:
            ret = {
                "ray_o": self.rays_o.cuda(),
                "ray_d": self.rays_d.cuda(),
                "ray_o_unseen": self.rays_o_unseen.cuda() if self.rays_o_unseen is not None else None,
                "ray_d_unseen": self.rays_d_unseen.cuda() if self.rays_d_unseen is not None else None,
                "camera_unseen": self.camera_unseen.cuda() if self.camera_unseen is not None else None,
                "depth_range": self.depth_range.cuda(),
                "camera": self.camera.cuda(),
                "rgb": self.rgb.cuda() if self.rgb is not None else None,
                "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                "src_rgbs_cond2": self.src_rgbs_cond2.cuda() if self.src_rgbs_cond2 is not None else None,
                "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            }
        else:
            ret = {
                "ray_o": self.rays_o.cuda(),
                "ray_d": self.rays_d.cuda(),
                "depth_range": self.depth_range.cuda(),
                "camera": self.camera.cuda(),
                "rgb": self.rgb.cuda() if self.rgb is not None else None,
                "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                "src_rgbs_cond2": self.src_rgbs_cond2.cuda() if self.src_rgbs_cond2 is not None else None,
                "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            }
        return ret



    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        # If the specified sample mode is "center", sample pixels from the center of the image
        if sample_mode == "center":
            # Calculate the border size around the center of the image
            border_H = int(self.H * (1 - center_ratio) / 2.0)
            border_W = int(self.W * (1 - center_ratio) / 2.0)

            # Generate a grid of pixel coordinates for the center region of the image
            u, v = np.meshgrid(
                np.arange(border_H, self.H - border_H), np.arange(border_W, self.W - border_W)
            )
            # Flatten the arrays of pixel coordinates
            u = u.reshape(-1)
            v = v.reshape(-1)

            # Select a random subset of the pixel coordinates
            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            # Calculate the corresponding pixel indices
            select_inds = v[select_inds] + self.W * u[select_inds]

        # If the specified sample mode is "uniform", sample pixels uniformly from the entire image
        elif sample_mode == "uniform":
            # Select a random subset of the pixel indices
            # print("H*W: ", self.H * self.W)
            select_inds = rng.choice(self.H * self.W, size=(N_rand,), replace=False)
        # If the specified sample mode is not recognized, raise an exception
        else:
            raise Exception("unknown sample mode!")

        # Return the selected pixel indices
        return select_inds

    def sample_random_patch(self, patch_size, center_ratio=0.9):
        border_H = int(self.H * (1 - center_ratio) / 2.0 - patch_size/2)
        border_W = int(self.W * (1 - center_ratio) / 2.0 - patch_size/2)

        # pixel coordinates
        u, v = np.meshgrid(
            np.arange(border_H, self.H - border_H), np.arange(border_W, self.W - border_W)
        )
        u = u.reshape(-1)
        v = v.reshape(-1)
        # print("u shape: ", u.shape)
        # print("v shape: ", v.shape)
        rand_center = rng.choice(u.shape[0], size=(1,), replace=False)
        select_inds = np.arange(rand_center - int(patch_size / 2), rand_center + int(patch_size / 2))
        select_inds = v[select_inds] + self.W * u[select_inds]
        # print("select inds shape: ", select_inds.shape)


        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        """
        :param N_rand: number of rays to be casted
        :return:
        """
        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)
        # print("Num selected indices: ", select_inds.shape)
        # print("N rand: ", N_rand)
        # select_inds = self.sample_random_patch(10)
        # sys.exit("EXIT in random_sample sample_ray.py")
        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        # Add unseen rays
        rays_o_unseen = self.rays_o_unseen
        rays_d_unseen = self.rays_d_unseen
        # print("rays_self.rays_o", self.rays_o_unseen[:5, :])
        # print("rays_self.rays_d", self.rays_d_unseen[:5, :])
        #
        # print("Part 2")
        # print("rays_self.rays_o", self.rays_o_unseen[1000:1000+5, :])
        # print("rays_self.rays_d", self.rays_d_unseen[10000:1000+5, :])
        #
        # print("Part 3")
        # print("rays_self.rays_o", self.rays_o_unseen[-5:, :])
        # print("rays_self.rays_d", self.rays_d_unseen[-5:, :])

        # rays_o_unseen = self.rays_o
        # rays_d_unseen = self.rays_d
        # print("select_inds max: ", max(select_inds))

        if self.rgb is not None:
            # print("self.rgb shape: ", self.rgb.shape)
            # print("self.rgb shape test 1: ", self.rgb[0, :])
            # print("self.rgb shape test 2: ", self.rgb[0])

            rgb = self.rgb[select_inds]
        else:
            rgb = None

        if self.rgb_cond2 is not None:
            # print("self.rgb_cond2 shape", self.rgb_cond2.shape)
            rgb_cond2 = self.rgb_cond2[select_inds]
        else:
            rgb_cond2 = None

        # print("self.src_rgbs in ray sampler shape: ", self.src_rgbs.shape)
        # sys.exit("Exit in ray sampler!")
        if self.camera_unseen is not None:
            ret = {
                "ray_o": rays_o.cuda(),
                "ray_d": rays_d.cuda(),
                "ray_o_unseen": rays_o_unseen.cuda(),
                "ray_d_unseen": rays_d_unseen.cuda(),
                "camera_unseen": self.camera_unseen.cuda(),
                "camera": self.camera.cuda(),
                "depth_range": self.depth_range.cuda(),
                "rgb": rgb.cuda() if rgb is not None else None,
                "rgb_cond2": rgb_cond2.cuda() if rgb_cond2 is not None else None,
                "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                "src_rgbs_cond2": self.src_rgbs_cond2.cuda() if self.src_rgbs_cond2 is not None else None,
                "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
                "selected_inds": select_inds,
            }
        else:
            ret = {
                "ray_o": rays_o.cuda(),
                "ray_d": rays_d.cuda(),
                "camera": self.camera.cuda(),
                "depth_range": self.depth_range.cuda(),
                "rgb": rgb.cuda() if rgb is not None else None,
                "rgb_cond2": rgb_cond2.cuda() if rgb_cond2 is not None else None,
                "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
                "src_rgbs_cond2": self.src_rgbs_cond2.cuda() if self.src_rgbs_cond2 is not None else None,
                "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
                "selected_inds": select_inds,
            }
        return ret
