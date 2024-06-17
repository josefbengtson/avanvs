import torch
import os
from gnt.transformer_network import GNT
from gnt.feature_network import ResUNet, ClassifyLatent
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import sys

def de_parallel(model):
    return model.module if hasattr(model, "module") else model


########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


def rename_key(d, old_key, new_key):
    if old_key not in d:
        return d
    new_dict = OrderedDict()
    for key, value in d.items():
        if key == old_key:
            key = new_key
        new_dict[key] = value
    return new_dict


class GNTModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True, appearance_change=False, pre_trained=False, out_folder=""):
        self.args = args
        self.appearance_change = appearance_change
        self.pre_trained = pre_trained
        self.load_z = args.load_z == 1


        z_dims_string = args.latent_dims
        z_dims_array_string = z_dims_string[1:-1]  # remove brackets
        self.z_dims = np.array(z_dims_array_string.split(',')).astype(int)

        input_latent_size = self.z_dims[0]
        self.z_conds = generate_z_conds(self.load_z, out_folder, input_latent_size)

        device = torch.device("cuda:{}".format(args.local_rank))

        # create coarse GNT
        self.net_coarse = GNT(
            args,
            z_conds=self.z_conds,
            in_feat_ch=self.args.coarse_feat_dim,
            posenc_dim=3 + 3 * 2 * 10,
            viewenc_dim=3 + 3 * 2 * 10,
            ret_alpha=args.N_importance > 0,
            appearance_change=appearance_change,
            z_dims=self.z_dims,
            out_folder=out_folder
        ).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,
            fine_out_ch=self.args.fine_feat_dim,
            single_net=self.args.single_net,
        ).to(device)

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())

        if input_latent_size == 2:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                ],
                lr=args.lrate_gnt,
            )
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.net_coarse.parameters()},
                    {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
                    {"params": self.z_conds}
                ],
                lr=args.lrate_gnt,
            )


        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor
        )

        out_folder = os.path.join(args.rootdir, "out", args.expname)
        self.start_step = self.load_from_ckpt(
            out_folder, load_opt=load_opt, load_scheduler=load_scheduler
        )

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse, device_ids=[args.local_rank], output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net, device_ids=[args.local_rank], output_device=args.local_rank
            )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()

    def save_model(self, filename):

        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }


        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location="cuda:{}".format(self.args.local_rank))
        else:
            to_load = torch.load(filename)
        loaded_keys = to_load["net_coarse"].keys()
        for key in loaded_keys:
            # if key == "view_selftrans.7.attn.v_fc.weight" and self.pre_trained and self.appearance_change:
            need_to_change_v_fc = False
            if need_to_change_v_fc:
                if key[0:14] == "view_selftrans" and key[17:] == "attn.v_fc.weight" and self.pre_trained and self.appearance_change:
                    intermediate_latent_size = self.z_dims[-1]
                    random_values = torch.empty(64, intermediate_latent_size, device='cuda')

                    nn.init.kaiming_normal_(random_values, mode='fan_in', nonlinearity='relu')
                    layer_nr = int(key[15])
                    if layer_nr == 7:
                        layer_name = "view_selftrans.{}.attn.v_fc.weight".format(layer_nr)
                        extended_tensor = torch.cat((to_load["net_coarse"][layer_name],
                                                     random_values), dim=1)
                        to_load["net_coarse"][layer_name] = extended_tensor

                    new_key = "ray_trans"+key[14:]
                    to_load["net_coarse"] = rename_key(to_load["net_coarse"], key, new_key)


            elif key[0:14] == "view_selftrans":
                new_key = "ray_trans"+key[14:]
                to_load["net_coarse"] = rename_key(to_load["net_coarse"], key, new_key)
            elif key[0:15] == "view_crosstrans":
                new_key = "view_trans"+key[15:]
                to_load["net_coarse"] = rename_key(to_load["net_coarse"], key, new_key)


        load_opt = False
        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])

        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        # Filter out z_cond1 and z_cond2
        model_dict = self.net_coarse.state_dict()
        pretrained_dict = {k: v for k, v in to_load["net_coarse"].items() if k in model_dict}
        model_dict.update(pretrained_dict)

        self.net_coarse.load_state_dict(model_dict)
        self.feature_net.load_state_dict(to_load["feature_net"])


    def load_from_ckpt(
        self, out_folder, load_opt=True, load_scheduler=True, force_latest_ckpt=False
    ):
        """
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        """

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [
                os.path.join(out_folder, f)
                for f in sorted(os.listdir(out_folder))
                if f.endswith(".pth")
            ]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            print("checkpoint path: ", ckpts)
            print("self.args.ckpt_path: ", self.args.ckpt_path)
            step = 0

        return step


def generate_z_conds(load_z, out_folder, latent_size):

    if load_z:
        folder_name = "z136_learned_CARLA_ONLY"
        # Load z from folder
        load_folder_z_cond1 = "latents/{}/z_cond1.npy".format(folder_name)
        z_cond_1 = np.load(load_folder_z_cond1)
        print("--------------------- Loading Z Conditions ---------------------")
        print("z cond 1 loaded: ", z_cond_1)

        load_folder_z_cond2 = "latents/{}/z_cond2.npy".format(folder_name)
        z_cond_2 = np.load(load_folder_z_cond2)
        print("z cond 2 loaded: ", z_cond_2)

        load_folder_z_cond3 = "latents/{}/z_cond3.npy".format(folder_name)
        z_cond_3 = np.load(load_folder_z_cond3)
        print("z cond 3 loaded: ", z_cond_3)

        load_folder_z_cond4 = "latents/{}/z_cond4.npy".format(folder_name)
        z_cond_4 = np.load(load_folder_z_cond4)
        print("z cond 4 loaded: ", z_cond_4)

        z_cond1 = torch.from_numpy(z_cond_1).float().to("cuda:0").requires_grad_()
        z_cond2 = torch.from_numpy(z_cond_2).float().to("cuda:0").requires_grad_()
        z_cond3 = torch.from_numpy(z_cond_3).float().to("cuda:0").requires_grad_()
        z_cond4 = torch.from_numpy(z_cond_4).float().to("cuda:0").requires_grad_()
        z_conds = [z_cond1, z_cond2, z_cond3, z_cond4]
    else:
        # Initialize z

        if latent_size == 2:
            # Fixed
            print("Generating Fixed initial latent codes")
            z_cond1 = torch.tensor([-1., 0]).to("cuda:0")
            z_cond2 = torch.tensor([0., 0]).to("cuda:0")
            z_cond3 = torch.tensor([0., 1]).to("cuda:0")
            z_cond4 = torch.tensor([1., 0]).to("cuda:0")
        else:
            # Randomly
            z_cond1 = nn.Parameter(torch.randn(latent_size).to(device='cuda').requires_grad_())
            z_cond2 = nn.Parameter(torch.randn(latent_size).to(device='cuda').requires_grad_())
            z_cond3 = nn.Parameter(torch.randn(latent_size).to(device='cuda').requires_grad_())
            z_cond4 = nn.Parameter(torch.randn(latent_size).to(device='cuda').requires_grad_())

            print("---------------------  Generating Z Conditions ---------------------")
            print("z cond 1 generated: ", z_cond1)
            print("z cond 2 generated: ", z_cond2)
            print("z cond 3 generated: ", z_cond3)
            print("z cond 4 generated: ", z_cond4)

        save_initial = False
        if save_initial:
            os.makedirs(out_folder + "/latents/", exist_ok=True)



            # print("cond 1 nr: ", cond1_nr)
            filename_txt_cond1 = out_folder + "/latents/z_cond1_initial.txt"
            filename_np_cond1 = out_folder + "/latents/z_cond1_initial"
            z_cond1_np = z_cond1.cpu().detach().numpy()
            np.savetxt(filename_txt_cond1, z_cond1_np)
            np.save(filename_np_cond1, z_cond1_np)

            filename_txt_cond2 = out_folder + "/latents/z_cond2_initial.txt"
            filename_np_cond2 = out_folder + "/latents/z_cond2_initial"
            z_cond2_np = z_cond2.cpu().detach().numpy()
            np.savetxt(filename_txt_cond2, z_cond2_np)
            np.save(filename_np_cond2, z_cond2_np)

            filename_txt_cond3 = out_folder + "/latents/z_cond3_initial.txt"
            filename_np_cond3 = out_folder + "/latents/z_cond3_initial"
            z_cond3_np = z_cond3.cpu().detach().numpy()
            np.save(filename_np_cond3, z_cond3_np)
            np.savetxt(filename_txt_cond3, z_cond3_np)

            filename_txt_cond4 = out_folder + "/latents/z_cond4_initial.txt"
            filename_np_cond4 = out_folder + "/latents/z_cond4_initial"
            z_cond4_np = z_cond4.cpu().detach().numpy()
            np.save(filename_np_cond4, z_cond4_np)
            np.savetxt(filename_txt_cond4, z_cond4_np)

        z_conds = [z_cond1, z_cond2, z_cond3, z_cond4]
    return z_conds
