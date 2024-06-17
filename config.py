import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--rootdir",
        type=str,
        default="./",
        help="the path to the project root directory. Replace this path with yours!",
    )

    parser.add_argument(
        "--out_rootdir",
        type=str,
        default="./",
        help="the path to the output root directory. Replace this path with yours!",
    )

    parser.add_argument("--expname", type=str, help="experiment name")
    parser.add_argument("--distributed", action="store_true", help="if use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="rank for distributed training")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 8)",
    )

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="ibrnet_collected",
        help="the training dataset, should either be a single dataset, "
        'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces',
    )
    parser.add_argument(
        "--dataset_weights",
        nargs="+",
        type=float,
        default=[],
        help="the weights for training datasets, valid when multiple datasets are used.",
    )
    parser.add_argument(
        "--train_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of training scenes from training dataset",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="llff_test", help="the dataset to evaluate"
    )
    parser.add_argument(
        "--eval_scenes",
        nargs="+",
        default=[],
        help="optional, specify a subset of scenes from eval_dataset to evaluate",
    )
    parser.add_argument(
        "--render_folder",
        nargs="+",
        default="carla_render",
        help="render source folder path",
    )

    ## others
    parser.add_argument(
        "--testskip",
        type=int,
        default=8,
        help="will load 1/N images from test/val sets, "
        "useful for large datasets like deepvoxels or nerf_synthetic",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Default"],
        help="Choose which conditions to load "
    )

    parser.add_argument(
        "--unseen_patch_size",
        type=int,
        default=5,
        help="Size of unseen patches"
        "Size of unseen patches",
    )


    ########## model options ##########
    ## ray sampling options
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="uniform",
        help="how to sample pixels from images for training:" "uniform|center",
    )
    parser.add_argument(
        "--center_ratio", type=float, default=0.8, help="the ratio of center crop to keep"
    )
    parser.add_argument(
        "--N_rand",
        type=int,
        default=32 * 16,
        help="batch size (number of random rays per gradient step)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 4,
        help="number of rays processed in parallel, decrease if running out of memory",
    )

    ## model options
    parser.add_argument(
        "--coarse_feat_dim", type=int, default=32, help="2D feature dimension for coarse level"
    )
    parser.add_argument(
        "--fine_feat_dim", type=int, default=32, help="2D feature dimension for fine level"
    )
    parser.add_argument(
        "--num_source_views",
        type=int,
        default=10,
        help="the number of input source views for each target view",
    )
    parser.add_argument(
        "--rectify_inplane_rotation", action="store_true", help="if rectify inplane rotation"
    )
    parser.add_argument("--coarse_only", action="store_true", help="use coarse network only")
    parser.add_argument(
        "--anti_alias_pooling", type=int, default=1, help="if use anti-alias pooling"
    )
    parser.add_argument("--trans_depth", type=int, default=4, help="number of transformer layers")
    parser.add_argument("--netwidth", type=int, default=64, help="network intermediate dimension")
    parser.add_argument(
        "--single_net",
        type=bool,
        default=True,
        help="use single network for both coarse and/or fine sampling",
    )

    parser.add_argument("--depth_weight", type=float, default=0, help="Weight for depth loss term")

    parser.add_argument("--q_weight", type=float, default=0, help="Weight for q loss term")
    parser.add_argument("--attention_loss_weight", type=float, default=0, help="Weight for attention loss term")
    parser.add_argument("--into_day_weight", type=float, default=1, help="Weight for attention loss term")

    parser.add_argument(
        "--useAppearanceChange",
        type=int,
        default=1,
        help="Include appearance change loss term",
    )
    parser.add_argument(
        "--show_interpol",
        type=int,
        default=1,
        help="Show interpolation in evaluation",
    )
    parser.add_argument(
        "--regularize_unseen",
        type=int,
        default=0,
        help="Regularize unseen patches",
    )

    parser.add_argument(
        "--latent_size",
        type=int,
        default=16,
        help="Size of latent variable",
    )


    parser.add_argument(
        "--latent_dims",
        type=str,
        default="[2]",
        help="List of latent dimensions",
    )

    parser.add_argument(
        "--condition_weights",
        type=str,
        default="[1,1,1,1]",
        help="List of condition weights",
    )
    parser.add_argument(
        "--depth_condition",
        type=int,
        default=0,
        help="Depth condition loss",
    )
    ######### Apearance Change ###########
    parser.add_argument(
        "--appearance_change",
        type=bool,
        default=False,
        help="Decide if appearance should be changed",
    )

    parser.add_argument(
        "--load_z",
        type=int,
        default=0,
        help="Decide if z should be loaded",
    )

    parser.add_argument(
        "--render_interpolation",
        type=int,
        default=0,
        help="Perform interpolation between latents when rendering",
    )

    parser.add_argument(
        "--from_real",
        type=int,
        default=1,
        help="Perform interpolation from",
    )
    
    parser.add_argument(
        "--alpha_render", type=float, default=0, help="Choose alpha for rendering"
    )

    ########## checkpoints ##########
    parser.add_argument(
        "--no_reload", action="store_true", help="do not reload weights from saved ckpt"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="specific weights npy file to reload for coarse network",
    )
    parser.add_argument(
        "--no_load_opt", action="store_true", help="do not load optimizer when reloading"
    )
    parser.add_argument(
        "--no_load_scheduler", action="store_true", help="do not load scheduler when reloading"
    )

    ########### iterations & learning rate options ##########
    parser.add_argument("--n_iters", type=int, default=250000, help="num of iterations")
    parser.add_argument(
        "--lrate_feature", type=float, default=1e-3, help="learning rate for feature extractor"
    )
    parser.add_argument("--lrate_gnt", type=float, default=5e-4, help="learning rate for gnt")
    parser.add_argument(
        "--lrate_decay_factor",
        type=float,
        default=0.5,
        help="decay learning rate by a factor every specified number of steps",
    )
    parser.add_argument(
        "--lrate_decay_steps",
        type=int,
        default=50000,
        help="decay learning rate by a factor every specified number of steps",
    )

    ########## rendering options ##########
    parser.add_argument(
        "--N_samples", type=int, default=64, help="number of coarse samples per ray"
    )
    parser.add_argument(
        "--N_importance", type=int, default=64, help="number of important samples per ray"
    )

    parser.add_argument(
        "--N_views", type=int, default=20, help="number of rendering views"
    )

    parser.add_argument(
        "--inv_uniform", action="store_true", help="if True, will uniformly sample inverse depths"
    )
    parser.add_argument(
        "--det", action="store_true", help="deterministic sampling for coarse and fine samples"
    )
    parser.add_argument(
        "--white_bkgd",
        action="store_true",
        help="apply the trick to avoid fitting to white background",
    )
    parser.add_argument(
        "--render_stride",
        type=int,
        default=1,
        help="render with large stride for validation to save time",
    )

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help="frequency of terminal printout")
    parser.add_argument(
        "--i_img", type=int, default=500, help="frequency of tensorboard image logging"
    )
    parser.add_argument(
        "--i_weights", type=int, default=10000, help="frequency of weight ckpt saving"
    )

    ########## evaluation options ##########
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="will take every 1/N images as LLFF test set, paper uses 8",
    )

    return parser
