from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .spaces_dataset_render import *

from .nerf_synthetic import *
from .shiny import *
from .llff_render import *
from .cmu_render import *
from .cmu_test import *
from .carla_render import *
from .carla_test import *
from .carla_eval import *

dataset_dict = {
    "spaces": SpacesFreeDataset,
    "spaces_render": SpacesFreeDatasetRender,
    "google_scanned": GoogleScannedDataset,
    "realestate": RealEstateDataset,
    "deepvoxels": DeepVoxelsDataset,
    "nerf_synthetic": NerfSyntheticDataset,
    "llff": LLFFDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "llff_test": LLFFTestDataset,
    "llff_render": LLFFRenderDataset,
    "shiny": ShinyDataset,
    "cmu_render": CMURenderDataset,
    "cmu_test": CMUTestDataset,
    "carla_render": CarlaRenderDataset,
    "carla_test": CarlaTestDataset,
    "carla_eval": CarlaEvalDataset,
}
