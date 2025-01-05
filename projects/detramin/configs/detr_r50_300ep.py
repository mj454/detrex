from detrex.config import get_config
from .models.detr_r50 import model

dataloader = get_config("common/data/custom_dataloader.py").dataloader
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
# https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl
# https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl
train.init_checkpoint = "detectron2://COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl"
# train.init_checkpoint = "detectron2://new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl"
train.output_dir = "./output/detramin_r50_300ep"
train.max_iter = 100000
train.checkpointer=dict(period=100, max_to_keep=100)
train.eval_period = 100
# modify lr_multiplier
lr_multiplier.scheduler.milestones = [50000, 100000]

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 2
dataloader.train.total_batch_size = 2
