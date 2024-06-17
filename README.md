# Adjustable Visual Appearance for Generalizable Novel View Synthesis
[Josef Bengtson](https://www.chalmers.se/en/persons/bjosef/)<sup>1</sup>,
[David Nilsson](https://www.chalmers.se/personer/davini/)<sup>1</sup>,
[Che-Tsung Lin](https://www.chalmers.se/en/persons/chetsung/)<sup>1</sup>,
[Marcel Büsching](https://www.kth.se/profile/busching?l=en)<sup>2</sup>,
[Fredrik Kahl](https://fredkahl.github.io/)<sup>1</sup>

<sup>1</sup>Chalmers University of Technology, <sup>2</sup>KTH Royal Institute of Technology


[Project Page](https://ava-nvs.github.io/) | [Paper](https://arxiv.org/abs/2306.01344)

This repository is built based on GNT's [offical repository](https://github.com/VITA-Group/GNT)

## Introduction



We present a generalizable novel view synthesis method which enables modifying the visual appearance of an observed scene so rendered views match a target weather or lighting condition without any scene specific training or access to reference views at the target condition.

Our method is based on a pretrained generalizable transformer architecture and is fine-tuned on synthetically generated scenes under different appearance conditions. This allows for rendering novel views in a consistent manner for 3D scenes that were not included in the training set, along with the ability to (i) modify their appearance to match the target condition and (ii) smoothly interpolate between different conditions. Experiments on real and synthetic scenes show that our method is able to generate 3D consistent renderings while making realistic appearance changes, including qualitative and quantitative comparisons.


![teaser](assets/intro_new.svg
)

## Installation

Clone this repository:

```bash
git clone https://github.com/josefbengtson/avanvs.git
cd GNT/
```

The code is tested with python 3.8, cuda == 11.1, pytorch == 1.10.1. Additionally dependencies are found in the requirements.txt file.
## Datasets

The CARLA dataset is split into [training](https://drive.google.com/file/d/1-l5X4GCPiql1W09jJIWQdyuam3k2JZH3/view?usp=sharing) and [evaluation](https://drive.google.com/file/d/1k9imjro_ag5WDDLcMmukgqi0CBqQspk5/view?usp=sharing) scenes.
The datasets must be downloaded to a directory `data/` within the project folder and must follow the below organization. 
```bash
├──data/
    ├──carla_test/
    ├──carla_render/
    ├──conditions_list.npy
```
The method can also be applied on real scenes from the Spaces data, which can be downloaded by:
```bash
# Spaces dataset
git clone https://github.com/augmentedperception/spaces_dataset
```

## Usage

### Training

```bash
# Train on carla training scenes
# python3 train.py --config <config> --n_iters <num iterations> --i_img <logging frequency> --i_print <printing frequency> --ckpt_path <chkpts paths> --N_samples <samples per ray> --N_rand <rays per batch> --load_z  <load existing latent> --conditions <conditions> --eval_scenes <eval scene> --rootdir <root path> --expname <name> --latent_dims <latent dims>. 
# Example:
python3 train.py --config configs/gnt_carla.txt --n_iters 50000 --i_img 5000 --i_print 500 --ckpt_path chktpts/model_720000.pth --N_samples 64 --N_rand 128 --load_z 0 --conditions [1,2,3,4] --eval_scenes TEST_Scene122 --rootdir path/to/root --expname Training --latent_dims [2,16,68,136]
```
We start training from this  [GNT checkpoint](https://drive.google.com/file/d/1YvOJXa5eGpKgoMYcxC2ma7prB1n5UwRn/view)

### Pre-trained Models
To reuse pretrained models, download this [checkpoint](https://drive.google.com/file/d/1bPVG_rapXu0oQhlbPy0WQdw-ebjAI-LR/view?usp=sharing) and place in chkpts directory. Then proceed to evaluation or rendering. 

### Evaluation
To compute metrics on validation scenes.
```bash
# evaluate on carla_render scenes
#python3 eval.py --rootdir <root path> --config <config> --expname <name> --chunk_size <rays per batch> --run_val --N_samples <samples per ray> --eval_dataset <eval dataset> --ckpt_path <chkpts paths> --conditions <conditions> --latent_dims <latent dims> --render_stride <render downsampling factor> --show_interpol <show interpolation result>
# Example:
python3 eval.py --rootdir path/to/root --config configs/gnt_full.txt --expname Evaluation --chunk_size 4096 --run_val --N_samples 64 --eval_dataset carla_eval --ckpt_path chktpts/model_770000.pth --conditions [1,4] --latent_dims [2,16,68,136] --render_stride 1 --show_interpol 0
```

### Rendering

To render videos of smooth camera paths for the real forward-facing scenes.

```bash
# python3 render.py --rootdir <root path> --config <config> --eval_dataset <eval dataset> --render_folder <render folder name> --chunk_size <rays per batch> --N_samples <samples per ray> --ckpt_path <chkpts paths> --latent_dims <latent dims> --N_views <number views to render> --render_stride <render downsampling factor> --expname <name> --eval_scenes <scene to render> --render_interpolation <interpolate between conditions> --conditions <conditions>
# Example for syntehtic CARLA scenes:
python3 render.py --rootdir path/to/root --config configs/gnt_full.txt --eval_dataset carla_render --render_folder carla_render --chunk_size 4096 --N_samples 64 --ckpt_path chktpts/model_770000.pth --latent_dims [2,16,68,136] --N_views 20 --render_stride 1 --expname Render --eval_scenes Scene6_right --render_interpolation 1 --conditions [1,4]

# Example for real spaces scenes:
python3 render.py --rootdir path/to/root --config configs/gnt_full.txt --eval_dataset spaces_render --render_folder spaces_render --chunk_size 4096 --N_samples 64 --ckpt_path chktpts/model_770000.pth --latent_dims [2,16,68,136] --N_views 20 --render_stride 1 --expname Render --eval_scenes 050 --render_interpolation 1 --conditions [1,4]
```



## Cite this work

If you find our work/code implementation useful for your own research, please cite our paper.

```
@misc{bengtson2023adjustable,
  title         = {Adjustable Visual Appearance for Generalizable Novel View Synthesis}, 
  author        = {Josef Bengtson and David Nilsson and Che-Tsung Lin and Marcel Büsching and Fredrik Kahl},
  year          = {2023},
  eprint        = {2306.01344},
  archivePrefix = {arXiv},
}
```
