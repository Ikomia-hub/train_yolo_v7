# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
import copy
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from train_yolo_v7.ikutils import prepare_dataset, download_model
import sys
import argparse
import logging
import os
import random
import time
from pathlib import Path
import numpy as np
import torch.distributed as dist
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter

from train_yolo_v7.yolov7.utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, \
    init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from train_yolo_v7.yolov7.utils.torch_utils import ModelEMA, select_device, intersect_dicts, \
    torch_distributed_zero_first, is_parallel
from train_yolo_v7.yolov7.utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from train_yolo_v7.yolov7.train import train
from datetime import datetime

logger = logging.getLogger(__name__)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainYolov7Param(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Create models folder
        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        dataset_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset")
        os.makedirs(models_folder, exist_ok=True)
        os.makedirs(dataset_folder, exist_ok=True)

        self.cfg["dataset_folder"] = dataset_folder
        self.cfg["model_name"] = "yolov7"
        self.cfg["use_pretrained"] = True
        self.cfg["model_path"] = ""
        self.cfg["epochs"] = 10
        self.cfg["batch_size"] = 16
        self.cfg["train_imgsz"] = 640
        self.cfg["test_imgsz"] = 640
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["config"] = ""
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"

    def set_values(self, param_map):
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["use_pretrained"] = utils.strtobool(param_map["use_pretrained"])
        self.cfg["model_path"] = param_map["model_path"]
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["train_imgsz"] = int(param_map["train_imgsz"])
        self.cfg["test_imgsz"] = int(param_map["test_imgsz"])
        self.cfg["dataset_split_ratio"] = float(param_map["dataset_split_ratio"])
        self.cfg["config"] = param_map["config"]
        self.cfg["output_folder"] = param_map["output_folder"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainYolov7(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)

        # Create parameters class
        if param is None:
            self.set_param_object(TrainYolov7Param())
        else:
            self.set_param_object(copy.deepcopy(param))
        
        self.enable_mlflow(False)


    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        param = self.get_param_object()
        dataset_input = self.get_input(0)

        # Conversion from Ikomia dataset to YoloV5
        print("Preparing dataset...")
        dataset_yaml = prepare_dataset(dataset_input, param.cfg["dataset_folder"],
                                       param.cfg["dataset_split_ratio"])

        print("Collecting configuration parameters...")
        self.opt = self.load_config(dataset_yaml)

        # Call begin_task_run for initialization
        self.begin_task_run()

        print("Start training...")
        self.start_training()

        # Call end_task_run to finalize process
        self.end_task_run()

    def load_config(self, dataset_yaml):
        param = self.get_param_object()

        if len(sys.argv) == 0:
            sys.argv = ["ikomia"]
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--notest', action='store_true', help='only test final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default='runs/train', help='save to project/name')
        parser.add_argument('--entity', default=None, help='W&B entity')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
        parser.add_argument('--bbox_interval', type=int, default=-1,
                            help='Set bounding-box image logging interval for W&B')
        parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
        parser.add_argument('--artifact_alias', type=str, default="latest",
                            help='version of dataset artifact to be used')

        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "yolov7", "cfg", "training",
                                   param.cfg["model_name"] + ".yaml")

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            parser.set_defaults(**config)

        opt = parser.parse_args(args=[])
        opt.data = dataset_yaml

        # Override with GUI parameters
        if param.cfg["config"]:
            opt.hyp = param.cfg["config"]
        else:
            opt.hyp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov7", opt.hyp)

        models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        opt.weights = param.cfg["model_path"] if not param.cfg["use_pretrained"] else \
            os.path.join(models_folder, param.cfg["model_name"] + ".pt")
        if not os.path.isfile(opt.weights):
            if param.cfg["use_pretrained"]:
                download_model(param.cfg["model_name"], models_folder)
        opt.epochs = param.cfg["epochs"]
        opt.batch_size = param.cfg["batch_size"]
        opt.img_size = [param.cfg["train_imgsz"], param.cfg["test_imgsz"]]
        opt.project = param.cfg["output_folder"]
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        opt.name = str_datetime
        opt.tb_dir = str((Path(core.config.main_cfg["tensorboard"]["log_uri"]) / opt.name))
        opt.stop_train = False

        if sys.platform == 'win32':
            opt.workers = 0

        return opt

    def start_training(self):
        param = self.get_param_object()
        # Set DDP variables
        self.opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        self.opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
        set_logging(self.opt.global_rank)

        # Resume
        wandb_run = check_wandb_resume(self.opt)
        if self.opt.resume and not wandb_run:  # resume an interrupted run
            ckpt = self.opt.resume if isinstance(self.opt.resume,
                                                 str) else get_latest_run()  # specified or most recent path
            assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
            apriori = self.opt.global_rank, self.opt.local_rank
            with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
                self.opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
            self.opt.cfg, self.opt.weights, self.opt.resume, self.opt.batch_size, self.opt.global_rank, \
            self.opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
            logger.info('Resuming training from %s' % ckpt)
        else:
            # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
            self.opt.data, self.opt.cfg, self.opt.hyp = check_file(self.opt.data), check_file(self.opt.cfg), check_file(
                self.opt.hyp)  # check files
            assert len(self.opt.cfg) or len(self.opt.weights), 'either --cfg or --weights must be specified'
            self.opt.img_size.extend(
                [self.opt.img_size[-1]] * (2 - len(self.opt.img_size)))  # extend to 2 sizes (train, test)
            self.opt.name = 'evolve' if self.opt.evolve else self.opt.name
            self.opt.save_dir = increment_path(Path(self.opt.project) / self.opt.name,
                                               exist_ok=self.opt.exist_ok | self.opt.evolve)  # increment run

        # DDP mode
        self.opt.total_batch_size = self.opt.batch_size
        device = select_device(self.opt.device, batch_size=self.opt.batch_size)
        if self.opt.local_rank != -1:
            assert torch.cuda.device_count() > self.opt.local_rank
            torch.cuda.set_device(self.opt.local_rank)
            device = torch.device('cuda', self.opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert self.opt.batch_size % self.opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            self.opt.batch_size = self.opt.total_batch_size // self.opt.world_size

        # Hyperparameters
        with open(self.opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

        if not param.cfg["config"]:
            nbs = 64  # nominal batch size
            hyp["lr0"] = hyp["lr0"] / nbs * self.opt.batch_size

        tb_writer = SummaryWriter(self.opt.tb_dir)  # Tensorboard

        # Train
        logger.info(self.opt)
        if not self.opt.evolve:
            if self.opt.global_rank in [-1, 0]:
                prefix = colorstr('tensorboard: ')
                logger.info(
                    f"{prefix}Start with 'tensorboard --logdir {self.opt.project}', view at http://localhost:6006/")

            train(hyp, self.opt, device, self.on_epoch_end, tb_writer)

        # Evolve hyperparameters (optional)
        else:
            # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
            meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                    'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                    'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                    'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                    'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                    'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                    'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                    'box': (1, 0.02, 0.2),  # box loss gain
                    'cls': (1, 0.2, 4.0),  # cls loss gain
                    'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                    'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                    'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                    'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                    'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                    'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                    'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                    'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                    'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                    'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                    'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                    'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                    'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                    'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                    'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                    'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                    'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                    'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                    'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                    'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                    'paste_in': (1, 0.0, 1.0)}  # segment copy-paste (probability)

            with open(self.opt.hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)  # load hyps dict
                if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                    hyp['anchors'] = 3

            assert self.opt.local_rank == -1, 'DDP mode not implemented for --evolve'
            self.opt.notest, self.opt.nosave = True, True  # only test/save final epoch
            # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
            yaml_file = Path(self.opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
            if self.opt.bucket:
                os.system('gsutil cp gs://%s/evolve.txt .' % self.opt.bucket)  # download evolve.txt if exists

            for _ in range(300):  # generations to evolve
                if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                    # Select parent(s)
                    parent = 'single'  # parent selection method: 'single' or 'weighted'
                    x = np.loadtxt('evolve.txt', ndmin=2)
                    n = min(5, len(x))  # number of previous results to consider
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    if parent == 'single' or len(x) == 1:
                        # x = x[random.randint(0, n - 1)]  # random selection
                        x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                    elif parent == 'weighted':
                        x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                    # Mutate
                    mp, s = 0.8, 0.2  # mutation probability, sigma
                    npr = np.random
                    npr.seed(int(time.time()))
                    g = np.array([x[0] for x in meta.values()])  # gains 0-1
                    ng = len(meta)
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                    for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                        hyp[k] = float(x[i + 7] * v[i])  # mutate

                # Constrain to limits
                for k, v in meta.items():
                    hyp[k] = max(hyp[k], v[1])  # lower limit
                    hyp[k] = min(hyp[k], v[2])  # upper limit
                    hyp[k] = round(hyp[k], 5)  # significant digits

                # Train mutation
                results = train(hyp.copy(), self.opt, device, self.on_epoch_end, tb_writer)

                # Write mutation results
                print_mutation(hyp.copy(), results, yaml_file, self.opt.bucket)

    def stop(self):
        super().stop()
        self.opt.stop_train = True

    @staticmethod
    def conform_metrics(metrics):
        new_metrics = {}
        for tag in metrics:
            if "train/" in tag:
                val = metrics[tag].item()
            else:
                val = metrics[tag]

            tag = tag.replace(":", "-")
            new_metrics[tag] = val

        return new_metrics

    def on_epoch_end(self, metrics, epoch):
        # Step progress bar:
        self.emit_step_progress()
        metrics = self.conform_metrics(metrics)
        self.log_metrics(metrics, epoch)


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainYolov7Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_yolo_v7"
        self.info.short_description = "Train YOLOv7 object detection models."
        self.info.description = "This plugin proposes train on YOLOv7 object detection models."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arxiv"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolov7"
        # Keywords used for search
        self.info.keywords = "yolo, v7, object, detection, real-time, coco"

    def create(self, param=None):
        # Create process object
        return TrainYolov7(self.info.name, param)
