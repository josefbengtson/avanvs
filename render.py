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
from PIL import Image
import datetime

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
def render(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, "out", args.expname)
    print("device: ", device)
    print("device name: ",  torch.cuda.get_device_name(0))

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

    assert (args.eval_dataset == "llff_render") \
           or (args.eval_dataset == "cmu_render") \
           or (args.eval_dataset == "carla_render")\
           or (args.eval_dataset == "spaces_render"), ValueError("rendering mode available only for llff/cmu/carla datasets")

    print("args.eval_scenes: ", args.eval_scenes)
    if args.eval_dataset == "spaces_render":
        dataset = dataset_dict[args.eval_dataset](args, scenes=args.eval_scenes)
    else:
        dataset = dataset_dict[args.eval_dataset](args, scenes=args.eval_scenes)

    loader = DataLoader(dataset, batch_size=1)
    iterator = iter(loader)

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # Choose if change appearance
    appearance_change = True
    pre_trained = True
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    if args.render_interpolation == 1:
        use_interpolation = True
    else:
        use_interpolation = False

    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler,
        appearance_change=appearance_change, pre_trained=pre_trained
    )
    # create projector
    projector = Projector(device=device)

    indx = 0
    now = datetime.datetime.now()
    prev = datetime.datetime.now()
    print("Start Time =", now)
    predicted_conditions_1 = np.array([])
    predicted_conditions_2 = np.array([])
    print("Number Views to render: ", len(iterator))
    while True:
        try:
            data = next(iterator)
        except:
            print("Break at indx: ", indx)
            break
        if args.local_rank == 0:

            if args.eval_dataset == "spaces_render":
                scene_name = args.eval_scenes[0]

            else:
                scene_name = data["scene_name"][0]
            # Print data keys and scene name for first iteration (Same for all iterations)
            if indx == 0:
                print("data keys: ", data.keys())
                print("scene name: ", scene_name)

            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            alpha = args.alpha_render
            # Generate view with changed appearance
            predicted_conditions_1, predicted_conditions_2 = log_view(
                indx,
                args,
                model,
                tmp_ray_sampler,
                projector,
                render_stride=args.render_stride,
                prefix="val/" if args.run_val else "train/",
                out_folder=out_folder,
                ret_alpha=args.N_importance > 0,
                predicted_conditions_1=predicted_conditions_1,
                predicted_conditions_2=predicted_conditions_2,
                appearance_change=appearance_change,
                alpha=alpha,
                use_interpolation=use_interpolation,
                scene_name=scene_name
            )
            torch.cuda.empty_cache()
            indx += 1
            now = datetime.datetime.now()
            time_diff = now-prev
            prev = now
            print("Current Time =", now)
            print("Time to render image =", time_diff)

def save_rgb(rgb, file_name):
    rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 1)
    rgb = (255 * rgb).astype('uint8')
    pil_image = Image.fromarray(rgb)
    pil_image.save(file_name)

@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    render_stride=1,
    prefix="",
    out_folder="",
    ret_alpha=False,
    predicted_conditions_1=np.array([]),
    predicted_conditions_2=np.array([]),
    appearance_change=False,
    alpha=0,
    use_interpolation=False,
    scene_name="Void"
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
            featmaps_cond2 =[None, None]

        num_views = args.N_views
        from_real = args.from_real == 1
        if use_interpolation:
            if from_real:
                alpha =1-global_step/num_views
            else:
                alpha = global_step/num_views
            # print("Alpha: ", alpha)
        else:
            alpha = alpha
        print("--------------------------------------------")
        print("GLOBAL STEP: ", global_step)
        print("Alpha: ", alpha)

        # Render a new view
        conditions = ray_sampler.conditions
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
            alpha=alpha,
            conditions=conditions
        )

    # Extract rendered views
    rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
    if "rgb_cond1_latent2" in ret["outputs_coarse"].keys():
        rgb_coarse_cond1_to_cond2 = img_HWC2CHW(ret["outputs_coarse"]["rgb_cond1_latent2"].detach().cpu())

    if "rgb_cond2" in ret["outputs_coarse"].keys():
        rgb_coarse_cond2 = img_HWC2CHW(ret["outputs_coarse"]["rgb_cond2"].detach().cpu())
        rgb_coarse_cond2_to_cond1 = img_HWC2CHW(ret["outputs_coarse"]["rgb_cond2_latent1"].detach().cpu())

    else:
        rgb_coarse_cond2 = None

    H = rgb_coarse.shape[1]
    W = rgb_coarse.shape[2]
    w_cut = 0
    h_cut = 0

    # Save rendered views
    out_folder = out_folder + "/" + scene_name + "/"
    out_folder_cond1 = os.path.join(out_folder, "cond1/")
    os.makedirs(out_folder_cond1, exist_ok=True)
    filename = os.path.join(out_folder_cond1, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
    save_rgb(rgb_coarse, filename)

    if rgb_coarse_cond1_to_cond2 is not None:
        rgb_coarse_cond1_to_cond2 = rgb_coarse_cond1_to_cond2[:, h_cut:H - h_cut, w_cut:W - w_cut]
        out_folder_cond1_to_cond2 = os.path.join(out_folder, "cond1_to_cond2/")
        os.makedirs(out_folder_cond1_to_cond2, exist_ok=True)
        filename = os.path.join(out_folder_cond1_to_cond2, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
        save_rgb(rgb_coarse_cond1_to_cond2, filename)

    if rgb_coarse_cond2 is not None:
        rgb_coarse_cond2 = rgb_coarse_cond2[:, h_cut:H - h_cut, w_cut:W - w_cut]
        out_folder_cond2 = os.path.join(out_folder, "cond2/")
        os.makedirs(out_folder_cond2, exist_ok=True)
        filename = os.path.join(out_folder_cond2, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
        save_rgb(rgb_coarse_cond2, filename)

        rgb_coarse_cond2_to_cond1 = rgb_coarse_cond2_to_cond1[:, h_cut:H - h_cut, w_cut:W - w_cut]
        out_folder_cond2_to_cond1 = os.path.join(out_folder, "cond2_to_cond1/")
        os.makedirs(out_folder_cond2_to_cond1, exist_ok=True)
        filename = os.path.join(out_folder_cond2_to_cond1, prefix[:-1] + "_{:03d}_coarse.png".format(global_step))
        save_rgb(rgb_coarse_cond2_to_cond1, filename)

    return predicted_conditions_1, predicted_conditions_2

if __name__ == "__main__":
    parser = config.config_parser()
    parser.add_argument("--run_val", action="store_true", help="run on val set")
    args = parser.parse_args()
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    render(args)
