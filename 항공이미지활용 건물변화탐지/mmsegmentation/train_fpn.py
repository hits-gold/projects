from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = '/home/work/maicon/baseline_ook/mmsegmentation/configs/segmenter/segmenter_vit-l_mask_8x1_640x640_160k_ade20k.py'
checkpoint_file = '/home/work/maicon/baseline_ook/mmsegmentation/checkpoints/segmenter_vit-l_mask_8x1_640x640_160k_ade20k_20220614_024513-4783a347.pth'

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config

from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import os.path as osp
import mmcv

classes = ('background', 'value1', 'value2', 'value3')
 

@DATASETS.register_module()
class aaDataset(CustomDataset):
  CLASSES = classes
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None



cfg = Config.fromfile(config_file)

cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg

cfg.model.decode_head.num_classes = 3

cfg.dataset_type = 'aaDataset'
cfg.data_root = '/home/work/maicon/data'

cfg.data.train.type = 'aaDataset'
cfg.data.train.data_root = '/home/work/maicon/data/train'
cfg.data.train.img_dir = 'x'
cfg.data.train.ann_dir = 'y'
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = 'aaDataset'
cfg.data.val.data_root = '/home/work/maicon/data/train'
cfg.data.val.img_dir = 'x'
cfg.data.val.ann_dir = 'y'
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = 'aaDataset'
cfg.data.test.data_root = '/home/work/maicon/data/train'
cfg.data.test.img_dir = 'x'
cfg.data.test.ann_dir = 'y'
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

cfg.load_from = checkpoint_file

# Set up working dir to save files and logs.
cfg.work_dir = '/home/work/maicon/baseline_ook/mmsegmentation/save_checkpoints'

cfg.runner.max_iteTrs = 200
cfg.log_config.interval = 50
cfg.evaluation.interval = 1000  # 모델 학습시 평가를 몇 번째 iteration마다 할 것인지 지정
cfg.checkpoint_config.interval = 1000  # 모델 학습시 학습한 모델을 몇 번째 iteration마다 저장할 것인지 지정

cfg.runner = dict(type='IterBasedRunner', max_iters=4000)  # Iteration으로 동작, Epoch로 동작하게 변경할 수도 있음
# cfg.runner = dict(type='EpochBasedRunner', max_epochs=4000)  # Epoch로 변경
cfg.workflow = [('train', 1)]

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

cfg.device = 'cuda'

cfg.model = dict(
    type='EncoderDecoder',
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth',
    backbone=dict(
        type='VisionTransformer',
        img_size=(640, 640),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        with_cls_token=True,
        interpolate_mode='bicubic'
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels=1024,
        num_classes=3,
        num_layers=2,
        num_heads=16,
        embed_dims=1024,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        norm_cfg=dict(type='BN', requires_grad=True)),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)))

    # Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict(CLASSES=classes, PALETTE=[[0, 0, 0], [128, 128, 0], [128, 64, 128], [0, 0, 0]]))
