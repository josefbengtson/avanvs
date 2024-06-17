import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader

from gnt.data_loaders import dataset_dict
from gnt.render_ray import render_rays
from gnt.render_image import render_single_image
from gnt.model import GNTModel
from gnt.sample_ray import RaySamplerSingleImage
from gnt.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, cycle, img2psnr, lpips, ssim

import config
import torch.distributed as dist
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset
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

def save_rgb(rgb, file_name):
    rgb = rgb.permute(1, 2, 0).detach().cpu().numpy()
    rgb = np.clip(rgb, 0, 1)
    rgb = (255 * rgb).astype('uint8')
    pil_image = Image.fromarray(rgb)
    pil_image.save(file_name)

def train(args):

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # Choose if change appearance
    appearance_change = True
    useChangeAppearanceLoss = args.useAppearanceChange == 1
    pre_trained = True
    showChangedAppearance = True

    additional_printing = True
    use_two_validations = True
    val_cond_1 = torch.tensor([1, 4])
    val_cond_2 = torch.tensor([2, 3])
    if appearance_change:
        print("Appearance Change Training is on")
    if useChangeAppearanceLoss:
        print("AppearanceChangeLoss loss is used")
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Weighting different conditions differently
    into_day_weight = args.into_day_weight # Specific weight when transforming into day


    device = "cuda:{}".format(args.local_rank)
    print("device: ", device)
    print("device name: ",  torch.cuda.get_device_name(0))

    out_folder = os.path.join(args.out_rootdir, "out", args.expname)
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

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)

    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        worker_init_fn=lambda _: np.random.seed(),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
    )

    # create validation dataset
    while True:
        val_dataset = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
        val_loader = DataLoader(val_dataset, batch_size=1)
        val_loader_iterator = iter(cycle(val_loader))
        val_data = next(val_loader_iterator)
        val_data_conditions = val_data["conditions"]
        union_conditions = np.union1d(val_cond_1, val_data_conditions)


        if len(union_conditions) == 2:
            break
        else:
            # print("In generate first validation set loop!")
            pass
    if use_two_validations:
        while True:
            val_dataset_2 = dataset_dict[args.eval_dataset](args, "validation", scenes=args.eval_scenes)
            val_loader_2 = DataLoader(val_dataset_2, batch_size=1)
            val_loader_iterator_2 = iter(cycle(val_loader_2))
            val_data_2 = next(val_loader_iterator_2)

            val_data_2_conditions = val_data_2["conditions"]
            union_conditions_2 = np.union1d(val_cond_2, val_data_2_conditions)

            if len(union_conditions_2) == 2:
                break
            else:
                pass
        print("Final val_data_conditions: ")
        print("val_data condtions: ", val_data["conditions"])
        print("val_data 2 condtions: ", val_data_2["conditions"])


    # Create GNT model
    model = GNTModel(
        args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler,
        appearance_change=appearance_change, pre_trained=pre_trained, out_folder=out_folder
    )
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    scalars_to_log = {}

    global_step = model.start_step + 1
    global_step_start = model.start_step + 1
    epoch = 0

    start = datetime.datetime.now()
    print("Start Time =", start)
    psnr_training = np.array([])
    training_steps = np.array([])
    psnr_validation = np.array([])
    lpips_validation_day_to_night = np.array([])
    lpips_validation_day_to_rain = np.array([])

    ssim_validation_day_to_night = np.array([])
    ssim_validation_day_to_rain = np.array([])
    lpips_training = np.array([])
    validation_steps = np.array([])

    # Start Training
    render_counter = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device, patch_size=args.unseen_patch_size)
            N_rand = int(
                1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
            )

            # Extract conditions from ray sampler
            if ray_sampler.conditions is not None:
                conditions = ray_sampler.conditions[0]
            else:
                conditions = None

            ray_batch = ray_sampler.random_sample(
                N_rand,
                sample_mode=args.sample_mode,
                center_ratio=args.center_ratio,
            )
            # Extract Feature Maps
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if ray_batch["src_rgbs_cond2"] is not None:
                featmaps_cond2 = model.feature_net(ray_batch["src_rgbs_cond2"].squeeze(0).permute(0, 3, 1, 2))

            else:
                featmaps_cond2 = [None, None]


            ret = render_rays(
                ray_batch=ray_batch,
                model=model,
                projector=projector,
                featmaps=featmaps,
                featmaps_cond2=featmaps_cond2,
                N_samples=args.N_samples,
                inv_uniform=args.inv_uniform,
                det=args.det,
                ret_alpha=args.N_importance > 0,
                appearance_change=appearance_change,
                conditions=conditions,
                save_latents=False
            )
            # compute loss
            model.optimizer.zero_grad()

            # Loss term from rgb color prediction
            loss_rgb, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log)
            if "rgb_cond2" in ret["outputs_coarse"].keys():
                loss_rgb_cond2, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log, key="rgb_cond2")
                if appearance_change:
                    loss_rgb_cond1_latent2, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log,
                                                                       key="rgb_cond1_latent2", key_gt="rgb_cond2")
                    loss_rgb_cond2_latent1, scalars_to_log = criterion(ret["outputs_coarse"], ray_batch, scalars_to_log,
                                                                       key="rgb_cond2_latent1", key_gt="rgb")
            else:
                loss_rgb_cond2 = 0
                loss_rgb_cond1_latent2 = 0
                loss_rgb_cond2_latent1 = 0

            # Re-weight if into day
            w_cond1_to_cond2 = 1
            w_cond2_to_cond1 = 1
            if conditions is not None:
                if conditions[1] == 1:
                    w_cond1_to_cond2 = into_day_weight
                if conditions[0] == 1:
                    w_cond2_to_cond1 = into_day_weight

            if global_step % args.i_print == 0 or global_step < 5:
                if additional_printing:
                    print("------------------------------")
                    print("------------------------------")
                    print("Global step: ", global_step)
                    if conditions is not None:
                        print("Conditions: ", conditions)
                        if w_cond1_to_cond2 != 1 or w_cond2_to_cond1 != 1:
                            print("Into day weight: ", into_day_weight)

                        print("rgb condition 1: ", round(float(loss_rgb), 3))
                        print("rgb condition 2: ", round(float(loss_rgb_cond2), 3))
                        if useChangeAppearanceLoss:
                            print("rgb condition 1 -> 2: ", round(float(loss_rgb_cond1_latent2), 3))
                            print("rgb condition 2 -> 1: ", round(float(loss_rgb_cond2_latent1), 3))
                        else:
                            print("Not using change appearance loss!")
                    else:
                        print("Non Carla scene")
                        print("rgb: ", round(float(loss_rgb), 3))
                    print("------------------------------")
                    print("------------------------------")


            if useChangeAppearanceLoss:
                loss = loss_rgb + loss_rgb_cond2 + \
                       w_cond1_to_cond2*loss_rgb_cond1_latent2 + \
                       w_cond2_to_cond1*loss_rgb_cond2_latent1

            else:
                loss = loss_rgb + loss_rgb_cond2


            loss.backward()
            scalars_to_log["loss"] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log["lr"] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0
            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret["outputs_coarse"]["rgb"], ray_batch["rgb"]).item()
                    scalars_to_log["train/coarse-loss"] = mse_error
                    scalars_to_log["train/coarse-psnr-training-batch"] = mse2psnr(mse_error)
                    psnr_training = np.append(psnr_training, mse2psnr(mse_error))
                    training_steps = np.append(training_steps, global_step)
                    if ret["outputs_fine"] is not None:
                        mse_error = img2mse(ret["outputs_fine"]["rgb"], ray_batch["rgb"]).item()
                        scalars_to_log["train/fine-loss"] = mse_error
                        scalars_to_log["train/fine-psnr-training-batch"] = mse2psnr(mse_error)

                    logstr = "{} Epoch: {}  step: {} ".format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += " {}: {:.6f}".format(k, scalars_to_log[k])
                    print(logstr)
                    print("each iter time {:.05f} seconds".format(dt))

                if global_step % args.i_img == 0:
                    print("Saving checkpoints at {} to {}...".format(global_step, out_folder))
                    now = datetime.datetime.now()
                    time_diff = now - start
                    print("Time since start: ", time_diff)
                    fpath = os.path.join(out_folder, "model_{:06d}.pth".format(global_step))
                    model.save_model(fpath)

                # if global_step % args.i_img == 0:
                num_include_at_start = 0
                if global_step % args.i_img == 0 or global_step<global_step_start+num_include_at_start:

                    print("Logging a random validation view...")
                    now = datetime.datetime.now()
                    time_diff = now - start
                    print("Time since start: ", time_diff)

                    val_data_one = next(val_loader_iterator)

                    # sys.exit("In validation spot")
                    tmp_ray_sampler_one = RaySamplerSingleImage(
                        val_data_one, device, render_stride=args.render_stride
                    )

                    H, W = tmp_ray_sampler_one.H, tmp_ray_sampler_one.W

                    if tmp_ray_sampler_one.rgb_cond2 is not None:
                        gt_img_cond2_one = tmp_ray_sampler_one.rgb_cond2.reshape(H, W, 3)
                    else:
                        gt_img_cond2_one = None

                    gt_img_cond1_one = tmp_ray_sampler_one.rgb.reshape(H, W, 3)

                    validation_steps = np.append(validation_steps, global_step)
                    # print("outfolder outer: ", out_folder)
                    print("Ray sampler one conditions: ", tmp_ray_sampler_one.conditions)

                    conditions_one = tmp_ray_sampler_one.conditions[0]


                    skip_validation = False
                    if not skip_validation:
                        psnr_validation, lpips_validation_day_to_night, ssim_validation_day_to_night = log_view(
                            global_step,
                            args,
                            model,
                            tmp_ray_sampler_one,
                            projector,
                            gt_img_cond1_one,
                            gt_img_cond2_one,
                            psnr_validation,
                            lpips_validation_day_to_night,
                            ssim_validation_day_to_night,
                            render_counter=render_counter,
                            render_stride=args.render_stride,
                            postfix="one",
                            out_folder=out_folder,
                            ret_alpha=args.N_importance > 0,
                            appearance_change=appearance_change,
                            showChangedAppearance=showChangedAppearance,
                            conditions=conditions_one
                        )

                    if use_two_validations:
                        val_data_two = next(val_loader_iterator_2)
                        tmp_ray_sampler_two = RaySamplerSingleImage(
                            val_data_two, device, render_stride=args.render_stride
                        )
                        print("Ray sampler two conditions: ", tmp_ray_sampler_two.conditions)
                        if tmp_ray_sampler_two.rgb_cond2 is not None:
                            gt_img_cond2_two = tmp_ray_sampler_two.rgb_cond2.reshape(H, W, 3)
                        else:
                            gt_img_cond2_two = None
                        gt_img_cond1_two = tmp_ray_sampler_two.rgb.reshape(H, W, 3)
                        conditions_two= tmp_ray_sampler_two.conditions[0]

                        psnr_validation, lpips_validation_day_to_rain, ssim_validation_day_to_rain = log_view(
                            global_step,
                            args,
                            model,
                            tmp_ray_sampler_two,
                            projector,
                            gt_img_cond1_two,
                            gt_img_cond2_two,
                            psnr_validation,
                            lpips_validation_day_to_rain,
                            ssim_validation_day_to_rain,
                            render_counter=render_counter,
                            render_stride=args.render_stride,
                            postfix="two",
                            out_folder=out_folder,
                            ret_alpha=args.N_importance > 0,
                            appearance_change=appearance_change,
                            showChangedAppearance=showChangedAppearance,
                            conditions=conditions_two
                        )
                    render_counter += 1

                    torch.cuda.empty_cache()


                    t_after_img = datetime.datetime.now()
                    print("Time to generate images: ", t_after_img-now)
                    print("-------------------------------------------------------------")
                    print("lpips validation cond1_to_cond2 one: ", lpips_validation_day_to_night)
                    print("lpips validation cond1_to_cond2 two: ", lpips_validation_day_to_rain)

                    # print("lpips validation: ", lpips_validation)

                    print("ssim validation cond1_to_cond2 two: ", ssim_validation_day_to_night)
                    print("ssim validation cond1_to_cond2 two: ", ssim_validation_day_to_rain)
                    print("validation steps: ", validation_steps)
                    print("-------------------------------------------------------------")

            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


@torch.no_grad()
def log_view(
    global_step,
    args,
    model,
    ray_sampler,
    projector,
    gt_img_cond1,
    gt_img_cond2,
    psnr=None,
    lpips_array=None,
    ssim_array=None,
    render_counter = 0,
    render_stride=1,
    postfix="",
    out_folder="",
    ret_alpha=False,
    appearance_change=False,
    showChangedAppearance=False,
    conditions=None,
):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net != None:
            featmaps = model.feature_net(ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))
            if "src_rgbs_cond2" in ray_batch.keys():
                if ray_batch["src_rgbs_cond2"] is not None:
                    featmaps_cond2 = model.feature_net(ray_batch["src_rgbs_cond2"].squeeze(0).permute(0, 3, 1, 2))
                else:
                    featmaps_cond2 = None
            else:
                featmaps_cond2 = [None, None]
        else:
            featmaps = [None, None]
            featmaps_cond2 = [None, None]

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
            out_folder=out_folder,
            global_step=global_step,
            conditions=conditions,
            save_latents=True
        )

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
            out_folder=out_folder,
            global_step=global_step,
            conditions=conditions,
            save_latents=True,
            alpha=0.5
        )

    print("out_folder pre: ", out_folder)
    if render_counter % 2:
        out_folder_cond1 = out_folder + "/cond1_even"
        out_folder_cond2 = out_folder + "/cond2_even"
    else:
        out_folder_cond1 = out_folder + "/cond1_odd"
        out_folder_cond2 = out_folder + "/cond2_odd"

    os.makedirs(out_folder_cond1, exist_ok=True)
    os.makedirs(out_folder_cond2, exist_ok=True)
    save_images(args, gt_img_cond1, ret["outputs_coarse"]["rgb"], render_stride, out_folder_cond1,
                "cond1",global_step, postfix=postfix)
    save_images(args, gt_img_cond2, ret["outputs_coarse"]["rgb_cond2"], render_stride, out_folder_cond2,
                "cond2",global_step, postfix=postfix)

    if showChangedAppearance:
        if render_counter % 2:
            out_folder_cond1_to_cond2 = out_folder + "/cond1_to_cond2_even"
            out_folder_cond2_to_cond1 = out_folder + "/cond2_to_cond1_even"
        else:
            out_folder_cond1_to_cond2 = out_folder + "/cond1_to_cond2_odd"
            out_folder_cond2_to_cond1 = out_folder + "/cond2_to_cond1_odd"
        os.makedirs(out_folder_cond1_to_cond2, exist_ok=True)
        os.makedirs(out_folder_cond2_to_cond1, exist_ok=True)

        lpips, ssim = save_images(args, gt_img_cond1, ret["outputs_coarse"]["rgb_cond1_latent2"],
                    render_stride, out_folder_cond1_to_cond2,"cond1_to_cond2", global_step, gt_img_cond2,
                    postfix=postfix, ret_alpha_50=ret_alpha_50["outputs_coarse"]["rgb_cond1_latent2"])
        lpips_array = np.append(lpips_array, lpips)
        ssim_array = np.append(ssim_array, ssim)

        save_images(args, gt_img_cond2, ret["outputs_coarse"]["rgb_cond2_latent1"],
                    render_stride, out_folder_cond2_to_cond1, "cond2_to_cond1", global_step, gt_img_cond1,
                    postfix=postfix, ret_alpha_50=ret_alpha_50["outputs_coarse"]["rgb_cond2_latent1"])

    model.switch_to_train()

    return psnr, lpips_array, ssim_array


def save_images(args, gt_img,rgb_conditions,render_stride,out_folder,
                prefix,global_step, gt_img_target=None, postfix="", ret_alpha_50=None):
    print("Generating validation images for input condition: ", prefix)
    gt_img_src = gt_img
    if gt_img_target is None:
        gt_img = gt_img
    else:
        gt_img = gt_img_target

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        gt_img_src = gt_img_src[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    gt_img_src = img_HWC2CHW(gt_img_src)
    rgb_pred = img_HWC2CHW(rgb_conditions.detach().cpu())
    if ret_alpha_50 is not None:
        rgb_pred_alpha = img_HWC2CHW(ret_alpha_50.detach().cpu())
    else:
        rgb_pred_alpha = None

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], gt_img_src.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], gt_img_src.shape[-1])
    showInterpolation = False
    if rgb_pred_alpha is not None and showInterpolation:
        rgb_im = torch.zeros(3, h_max, 4 * w_max)
        rgb_im[:, : gt_img_src.shape[-2], : gt_img_src.shape[-1]] = gt_img_src
        rgb_im[:, : rgb_gt.shape[-2], w_max: w_max + rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, : rgb_pred_alpha.shape[-2], 2 * w_max: 2 * w_max + rgb_pred_alpha.shape[-1]] = rgb_pred_alpha
        rgb_im[:, : rgb_pred.shape[-2], 3 * w_max: 3 * w_max + rgb_pred.shape[-1]] = rgb_pred
    else:
        rgb_im = torch.zeros(3, h_max, 3 * w_max)
        rgb_im[:, : gt_img_src.shape[-2], : gt_img_src.shape[-1]] = gt_img_src
        rgb_im[:, : rgb_gt.shape[-2], w_max: w_max + rgb_gt.shape[-1]] = rgb_gt
        rgb_im[:, : rgb_pred.shape[-2], 2 * w_max: 2 * w_max + rgb_pred.shape[-1]] = rgb_pred

    out_folder = os.path.join(out_folder, postfix)

    os.makedirs(out_folder, exist_ok=True)
    filename = out_folder + "/{:03d}.png".format(global_step)
    save_rgb(rgb_im, filename)

    pred_rgb = (
        rgb_conditions
    )
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    lpips_curr_img = lpips(pred_rgb, gt_img, format="HWC").item()
    ssim_curr_img = ssim(pred_rgb, gt_img, format="HWC").item()

    print(prefix + "psnr_image: ", psnr_curr_img)
    print(prefix + "lpips_image: ", lpips_curr_img)
    print(prefix + "ssim_image: ", ssim_curr_img)

    return lpips_curr_img, ssim_curr_img
if __name__ == "__main__":
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(args.local_rank)

    train(args)
