import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log, key="rgb", key_gt=None):
        """
        training criterion
        """
        if key_gt is None:
            key_gt = key

        pred_rgb = outputs[key]


        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch[key_gt]
        # print("pred_mask: ", pred_mask)
        #
        # print("pred_rgb: ", pred_rgb.shape)
        # print("gt_rgb: ", gt_rgb.shape)
        loss = img2mse(pred_rgb, gt_rgb, pred_mask)


        return loss, scalars_to_log

