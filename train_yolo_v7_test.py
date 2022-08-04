import logging
from ikomia.core import task, ParamMap
from ikomia.utils.tests import run_for_test
import os

logger = logging.getLogger(__name__)


def test(t, data_dict):
    logger.info("===== Test::train yolo v7 =====")
    input_dataset = t.getInput(0)
    params = task.get_parameters(t)
    params["epochs"] = 2
    params["batch_size"] = 1
    params["dataset_split_ratio"] = 0.5
    task.set_parameters(t, params)
    input_dataset.load(data_dict["datasets"]["detection"]["dataset_coco"])
    yield run_for_test(t)
