import argparse
import os
import urllib

from detectron2.data.detection_utils import read_image
from few_shot_detection.demo.predictor import VisualizationDemo
from few_shot_detection.fsdet.config import get_cfg

MODEL_NAME = "fsdet"


class FSDET(object):
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

    def import_data(self):
        filename = "dog.jpg"

        # Download an example image from the pytorch website
        if not os.path.isfile(filename):
            url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
            try:
                urllib.URLopener().retrieve(url, filename)
            except:
                urllib.request.urlretrieve(url, filename)

        img = read_image(filename, format="BGR")
        print(img.shape)

        return img, None

    def import_model(self):
        args = self.get_parser().parse_args(
            [
                "--config-file",
                "/scratch/wcheng/PipeSwitch/few_shot_detection/configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
                "--opts",
                "MODEL.WEIGHTS",
                "fsdet://coco/tfa_cos_1shot/model_final.pth",
            ]
        )
        cfg = self.setup_cfg(args)
        demo = VisualizationDemo(cfg)
        return demo.predictor

    def partition_model(self, model):
        group_list = [[child] for child in model.children()]
        return group_list
