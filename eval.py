import os
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_image import render_single_image
from gnt.model import GNTModel
from gnt.sample_ray import RaySamplerSingleImage
from utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
import config
import torch.distributed as dist
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset
from skimage.metrics import structural_similarity
from PIL import Image


def save_rgb(rgb, file_name):
    rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 1)
    rgb = (255 * rgb).astype('uint8')
    pil_image = Image.fromarray(rgb)
    pil_image.save(file_name)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@torch.no_grad()
def eval(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("outputs will be saved to {}".format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, "config.txt")
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    if args.run_val == False:
        # create training dataset
        dataset, sampler = create_training_dataset(args)
        # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
        # please use distributed parallel on multiple GPUs to train multiple target views per batch
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            worker_init_fn=lambda _: np.random.seed(),
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            # shuffle=True if sampler is None else False,
        )
        iterator = iter(loader)
    else:
        # create validation dataset
        print("args.eval_scenes: ", args.eval_scenes)
        print("args.eval_dataset: ", args.eval_dataset)
        # print("args.eval_scenes[0]: ", args.eval_scenes[0])
        # # Turn scene into string
        # scene = str(args.eval_scenes[0])
        # print("Scene: ", scene)
        dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)

        loader = DataLoader(dataset, batch_size=1)
        iterator = iter(loader)

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # Choose if change appearance
    appearance_change = True
    show_interpol = args.show_interpol == 1
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler, appearance_change=appearance_change
    )
    # create projector
    projector = Projector(device=device)

    indx = 0
    psnr_scores = np.empty((0, 4))
    lpips_scores = np.empty((0, 4))
    ssim_scores = np.empty((0, 4))
    while True:
        try:
            data = next(iterator)
        except:
            break
        if args.local_rank == 0:
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img_cond1 = tmp_ray_sampler.rgb.reshape(H, W, 3)
            gt_img_cond2 = tmp_ray_sampler.rgb_cond2.reshape(H, W, 3)
            scene_name = data["scene"][0]

            print("------------------------ Validation step {} -------------------------".format(indx))
            lpips_curr_img, ssim_curr_img,  psnr_curr_img = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                gt_img_cond1,
                gt_img_cond2,
                projector,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                ret_alpha=args.N_importance > 0,
                appearance_change=appearance_change,
                scene_name=scene_name,
                show_interpol=show_interpol
            )
            psnr_scores = np.append(psnr_scores, [psnr_curr_img], axis=0)
            ssim_scores = np.append(ssim_scores, [ssim_curr_img], axis=0)
            lpips_scores = np.append(lpips_scores, [lpips_curr_img], axis=0)

            torch.cuda.empty_cache()
            indx += 1

    conditions = tmp_ray_sampler.conditions[0]

    labels = ["condition {}:    ".format(conditions[0]), "condition {}:    ".format(conditions[1]),
              "condition {} to condition {}:    ".format(conditions[0], conditions[1]),
              "condition {} to condition {}:     ".format(conditions[1], conditions[0])]


    print("------------------------ Average Metrics -------------------------")
    print("------------------")
    print("Average PSNR: ")
    print("------------------")

    psnr_means = np.mean(psnr_scores, 0)
    for i in range(4):
        print(labels[i] + str(psnr_means[i]))
    print("------------------")
    print("Average SSIM: ")
    print("------------------")
    ssim_means = np.mean(ssim_scores, 0)
    for i in range(4):
        print(labels[i] + str(ssim_means[i]))
    print("------------------")
    print("Average LPIPS: ")
    print("------------------")
    lpips_means = np.mean(lpips_scores, 0)
    for i in range(4):
        print(labels[i] + str(lpips_means[i]))

def save_images(args, gt_img,rgb_conditions,render_stride,out_folder,prefix,
                global_step, gt_img_target=None, scene_name="Void", ret_alpha_50=None):
    gt_img_src = gt_img
    if gt_img_target is None:
        gt_img = gt_img
    else:
        gt_img = gt_img_target

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        gt_img_src = gt_img_src[::render_stride, ::render_stride]
    H = gt_img.shape[0]
    W = gt_img.shape[1]

    rgb_gt = img_HWC2CHW(gt_img)
    gt_img_src = img_HWC2CHW(gt_img_src)
    rgb_pred = img_HWC2CHW(rgb_conditions.detach().cpu())
    if ret_alpha_50 is not None:
        rgb_pred_alpha = img_HWC2CHW(ret_alpha_50.detach().cpu())
    else:
        rgb_pred_alpha = None


    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], gt_img_src.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], gt_img_src.shape[-1])

    if rgb_pred_alpha is None:
        rgb_im = torch.zeros(3, h_max, 3 * w_max)
        rgb_im[:, : gt_img_src.shape[-2], : gt_img_src.shape[-1]] = gt_img_src
        rgb_im[:, : rgb_gt.shape[-2], w_max: w_max + rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, : rgb_pred.shape[-2], 2 * w_max: 2 * w_max + rgb_pred.shape[-1]] = rgb_pred
    else:
        rgb_im = torch.zeros(3, h_max, 4 * w_max)
        rgb_im[:, : gt_img_src.shape[-2], : gt_img_src.shape[-1]] = gt_img_src
        rgb_im[:, : rgb_gt.shape[-2], w_max: w_max + rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, : rgb_pred_alpha.shape[-2], 2 * w_max: 2 * w_max + rgb_pred_alpha.shape[-1]] = rgb_pred_alpha
        rgb_im[:, : rgb_pred.shape[-2], 3 * w_max: 3 * w_max + rgb_pred.shape[-1]] = rgb_pred

    os.makedirs(out_folder, exist_ok=True)
    filename = out_folder + "/" + scene_name + "_{}.png".format(global_step)

    save_rgb(rgb_im, filename)

    pred_rgb = (
        rgb_conditions
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    (ssim_curr_img, diff) = structural_similarity(gt_img.numpy(), pred_rgb.numpy(), full=True, channel_axis=2, data_range=2, multichannel=True)
    return lpips_curr_img, ssim_curr_img, psnr_curr_img


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    gt_img_cond1,
    gt_img_cond2,
    projector,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    appearance_change=False,
    scene_name="Void",
    show_interpol=True
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if ray_batch["src_rgbs_cond2"] is not None:
                featmaps_cond2 = model.feature_net(ray_batch["src_rgbs_cond2"].squeeze(0).permute(0, 3, 1, 2))
            else:
                featmaps_cond2 = [None, None]
        else:
            featmaps = [None, None]
            featmaps_cond2 = [None, None]

        conditions = ray_sampler.conditions[0]
        ret = render_single_image(
            ray_sampler=ray_sampler,
            ray_batch=ray_batch,
            model=model,
            projector=projector,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            render_stride=render_stride,
            featmaps=featmaps,
            featmaps_cond2=featmaps_cond2,
            ret_alpha=ret_alpha,
            appearance_change=appearance_change,
            save_latents=False,
            conditions=conditions,
        )
        if show_interpol:
            ret_alpha_50 = render_single_image(
                ray_sampler=ray_sampler,
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                chunk_size=args.chunk_size,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                det=True,
                render_stride=render_stride,
                featmaps=featmaps,
                featmaps_cond2=featmaps_cond2,
                ret_alpha=ret_alpha,
                appearance_change=appearance_change,
                save_latents=False,
                conditions=conditions,
                alpha=0.5
            )
            ret_alpha_50_cond1_to_cond2 = ret_alpha_50["outputs_coarse"]["rgb_cond1_latent2"]
            ret_alpha_50_cond2_to_cond1 = ret_alpha_50["outputs_coarse"]["rgb_cond2_latent1"]
        else:
            ret_alpha_50_cond1_to_cond2 = None
            ret_alpha_50_cond2_to_cond1 = None


    out_folder_cond1 = out_folder + "/cond{}".format(int(conditions[0]))
    out_folder_cond2 = out_folder + "/cond{}".format(int(conditions[1]))

    out_folder_cond1_to_cond2 = out_folder + "/cond{}_to_cond_{}".format(int(conditions[0]), int(conditions[1]))
    out_folder_cond2_to_cond1 = out_folder + "/cond{}_to_cond_{}".format(int(conditions[1]), int(conditions[0]))

    os.makedirs(out_folder_cond1, exist_ok=True)
    os.makedirs(out_folder_cond2, exist_ok=True)
    os.makedirs(out_folder_cond1_to_cond2, exist_ok=True)
    os.makedirs(out_folder_cond2_to_cond1, exist_ok=True)

    lpips_array = np.zeros(4)
    ssim_array = np.zeros(4)
    psnr_array = np.zeros(4)
    lpips_array[0], ssim_array[0], psnr_array[0] = save_images(args, gt_img_cond1, ret["outputs_coarse"]["rgb"], render_stride, out_folder_cond1,
                prefix, global_step, gt_img_cond1,scene_name)
    lpips_array[1], ssim_array[1], psnr_array[1] = save_images(args, gt_img_cond2, ret["outputs_coarse"]["rgb_cond2"], render_stride, out_folder_cond2,
                prefix, global_step, gt_img_cond2,scene_name)

    lpips_array[2],  ssim_array[2], psnr_array[2] = save_images(args, gt_img_cond1, ret["outputs_coarse"]["rgb_cond1_latent2"], render_stride, out_folder_cond1_to_cond2,
                prefix, global_step, gt_img_cond2,scene_name,ret_alpha_50=ret_alpha_50_cond1_to_cond2)
    lpips_array[3],  ssim_array[3], psnr_array[3] = save_images(args, gt_img_cond2, ret["outputs_coarse"]["rgb_cond2_latent1"], render_stride, out_folder_cond2_to_cond1,
                prefix, global_step, gt_img_cond1,scene_name, ret_alpha_50=ret_alpha_50_cond2_to_cond1)

    labels = ["condition {}:    ".format(int(conditions[0])), "condition {}:    ".format(int(conditions[1])),
              "condition {} to condition {}:    ".format(int(conditions[0]), int(conditions[1])),
              "condition {} to condition {}:     ".format(int(conditions[1]), int(conditions[0]))]

    print("SCENE NAME: ", scene_name)
    print("--------------------------")
    print("PSNR:")
    print("--------------------------")
    for i in range(4):
        print(labels[i] + str(psnr_array[i]))
    print("--------------------------")
    print("SSIM:")
    print("--------------------------")
    for i in range(4):
        print(labels[i] + str(ssim_array[i]))
    print("--------------------------")
    print("LPIPS:")
    print("--------------------------")
    for i in range(4):
        print(labels[i] + str(lpips_array[i]))

    return lpips_array, ssim_array, psnr_array


if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    eval(args)


