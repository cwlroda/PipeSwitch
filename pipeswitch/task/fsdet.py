import argparse
import os
import json
from urllib import request
from redis import Redis
from pprint import pformat
import torch

from detectron2.data import (
    MetadataCatalog,
)
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T

from few_shot_detection.fsdet.checkpoint import DetectionCheckpointer
from few_shot_detection.fsdet.config import get_cfg
from few_shot_detection.fsdet.modeling import build_model

from pipeswitch.common.consts import REDIS_HOST, REDIS_PORT, Timers
from pipeswitch.profiling.timer import timer
from pipeswitch.common.logger import logger


MODEL_NAME = "fsdet"


class FSDET:
    def __init__(self):
        self.args = self.get_parser().parse_args(
            [
                "--config-file",
                "/scratch/wcheng/PipeSwitch/few_shot_detection/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
                "--opts",
                "MODEL.WEIGHTS",
                "fsdet://coco/tfa_cos_1shot/model_final.pth",
            ]
        )
        self._data_loader = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            encoding="utf-8",
            decode_responses=True,
        )

    @torch.no_grad()
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model
        """
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.transform_gen.get_transform(original_image).apply_image(
            original_image
        )
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.freeze()
        return cfg

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="FsDet demo for builtin models"
        )
        parser.add_argument(
            "--config-file",
            default=(
                "configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml"
            ),
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--webcam", action="store_true", help="Take inputs from webcam."
        )
        parser.add_argument("--video-input", help="Path to video file.")
        parser.add_argument(
            "--input",
            nargs="+",
            help=(
                "A list of space separated input images; "
                "or a single glob pattern such as 'directory/*.jpg'"
            ),
        )
        parser.add_argument(
            "--output",
            help=(
                "A file or directory to save output visualizations. "
                "If not given, will show output in an OpenCV window."
            ),
        )

        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help=(
                "Modify config options using the command-line 'KEY VALUE' pairs"
            ),
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser

    def import_data(self, task_key):
        data_str = self._data_loader.get(task_key)
        data = json.loads(data_str)
        logger.spam(f"\n{pformat(object=data, indent=1, width=1)}")
        img_url = data["task"]["items"][0]["urls"]["-1"]
        img_name = img_url.split("/")[-1]
        if not os.path.exists(img_name):
            request.urlretrieve(img_url, img_name)
        img = read_image(img_name, format="BGR")
        return img

    def import_model(self, device=None):
        cfg = self.setup_cfg(self.args)
        if device is not None:
            cfg.defrost()
            cfg.MODEL.DEVICE = device
            cfg.freeze()
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        return self.model

    def partition_model(self, model):
        group_list = [[child] for child in model.children()]
        return group_list
