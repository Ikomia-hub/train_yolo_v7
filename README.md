<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_yolo_v7/main/icons/icon.png" alt="Algorithm icon">
  <h1 align="center">train_yolo_v7</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_yolo_v7">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_yolo_v7">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_yolo_v7/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_yolo_v7.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train on YOLOv7 object detection models.

![Desk object detection](https://raw.githubusercontent.com/Ikomia-hub/train_yolo_v7/main/icons/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add train algorithm
train = wf.add_task(name="train_yolo_v7", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **train_imgsz** (int) - default '640': Size of the training image.
- **test_imgsz** (int) - default '640': Size of the eval image.
- **epochs** (int) - default '10': Number of complete passes through the training dataset.
- **batch_size** (int) - default '16': Number of samples processed before the model is updated.
- **dataset_split_ratio** (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- **output_folder** (str, *optional*): path to where the model will be saved. 
- **config_file** (str, *optional*): Path to hyperparameters configuration file .yaml. 

**Parameters** should be in **strings format**  when added to the dictionary.



```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "detection",
}) 

# Add train algorithm
train = wf.add_task(name="train_yolo_v7", auto_connect=True)
train.set_parameters({
    "batch_size": "2",
    "epochs": "5",
    "train_imgsz": "640",
    "test_imgsz": "640",
    "dataset_split_ratio": "0.9"
})

# Launch your training on your data
wf.run()
```

