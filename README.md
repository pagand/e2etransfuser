# DMFuser: Distilled Multi-Task Learning for  End-to-end Transformer-Based Sensor Fusion in Autonomous Driving



https://github.com/user-attachments/assets/7901c0d9-699d-4d0f-8cfb-f045c9200615


## Contents

1. [Setup](#setup)
2. [Dataset](#dataset)
3. [Training](#training)
3. [Evaluation](#evaluation)


## Setup

Clone the repo, setup CARLA 0.9.10.1, and build the conda environment:

```Shell
git clone git@github.com:pagand/e2etransfuser.git
cd e2etransfuser
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate tfuse
```
1-  if you have 10 <CUDA <=10.2
```Shell
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html 
```

2-  if you have CUDA >10.2
```Shell
pip uninstall torch torchvision torchaudio #(run twice)
pip install torch==1.12.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1%2Bcu113.html
pip install mmcv-full -f  https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
```

3- Alternatively
```Shell
pip uninstall torch torchvision torchaudio #(run twice)
```
3-1- Install the correct version of pytorch given your CUDA from [previous versions](https://pytorch.org/get-started/previous-versions/) or [start locally](https://pytorch.org/get-started/locally/). Replcae the {version} with the correct compatible version.
```Shell
conda install pytorch=={version1} torchvision=={version2} cudatoolkit={version3} -c pytorch
```
3-2- Install torch-scatter by finding your closest CUDA/pytroch version in [this address](https://data.pyg.org/whl/). Then replace the {address} with that.
```Shell
pip install torch-scatter -f {adderss}
```

3-3- Install mmcv-full acording to your pytorch and CUDA choose the correct prebuilt package available in [this address](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)


4- Install Huggingface transformers or follow [the link](https://huggingface.co/docs/transformers/installation)

```Shell
pip install transformers
```

## Dataset
Our dataset is generated via a privileged agent which we call the autopilot (`/transfuser_pami/team_code_autopilot/autopilot.py`) in 8 CARLA towns using the routes and scenario files provided in [this folder](./transfuser_pami/leaderboard/data/training/). See the [transfuser_pami/tools/dataset](./transfuser_pami/tools/dataset) folder for detailed documentation regarding the training routes and scenarios. 

The dataset is structured as follows:
```
- Scenario
    - Town
        - Route
            - rgb: camera images
            - depth: corresponding depth images
            - semantics: corresponding segmentation images
            - lidar: 3d point cloud in .npy format
            - topdown: topdown segmentation maps
            - label_raw: 3d bounding boxes for vehicles
            - measurements: contains ego-agent's position, velocity and other metadata
```


### Option 1: Data generation
We have provided the scripts for data generation that we used to train our autopilot agent. To generate data, the first step is to launch a CARLA server:

```Shell
cd transfuser_pami
./CarlaUE4.sh --world-port=2000 -opengl
```

For more information on running CARLA servers (e.g. on a machine without a display), see the [official documentation.](https://carla.readthedocs.io/en/stable/carla_headless/) Once the server is running, use the script below for generating training data:
```Shell
./leaderboard/scripts/datagen.sh <carla root> <working directory of this repo (*/transfuser/)>
```

The main variables to set for this script are `SCENARIOS` and `ROUTES`. 


### Option 2: Downloading dataset
A minimal dataset  (210GB) without the long scenario, where the camera is mounted in 2.3 m, can be downloaded by running:

```Shell
cd transfuser_pami
chmod +x download_data.sh
./download_data.sh
```

The data with camera mounted at 1.8 m, can be download from [HuggingFace dataset](https://huggingface.co/datasets/mmahdavi/carla_1.8m/tree/main).

### Dataset Augmentation

In order to generate long route and add to the current data, apply the data generation for long route. We have added [Town01long, Town02long, Town03long,Town04long, Town06long] for training and Town05long for validation.

To augment the vehicular control for the next n-step, use the script bellow.

```Shell
cd utilx
python augmentcontroldata.py
```

## TRAINING
You can train different baselines. For each method, follow the corresponding section. Check the config file in each folder accordingly. 

### DMFuser

Or you can upload the latest model path files from [this directory](https://huggingface.co/pagand/DMFuser/tree/main). 


### X13 
The model will be saved in a newly created folder log.
```Shell
cd x13
python train.py
```
To predict expert's driving records for task-wise evaluation
```Shell
python3 predict_expert.py is intended 
```


### Transfuser PAMI

The code for training via imitation learning is provided in [train.py.](./team_code_transfuser/train.py) \
A minimal example of running the training script on a single machine:
```Shell
cd transfuser_pami/team_code_transfuser
python train.py --batch_size 10 --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --parallel_training 0
```
The training script has many more useful features documented at the start of the main function. 
One of them is parallel training. 
The script has to be started differently when training on a multi-gpu node:
```Shell
cd team_code_transfuser
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d train.py --logdir /path/to/logdir --root_dir /path/to/dataset_root/ --parallel_training 1
```
Enumerate the GPUs you want to train on with CUDA_VISIBLE_DEVICES.
Set the variable OMP_NUM_THREADS to the number of cpus available on your system.
Set OPENBLAS_NUM_THREADS=1 if you want to avoid threads spawning other threads.
Set --nproc_per_node to the number of available GPUs on your node.

## Evaluation

### Longest6 benchmark
We make some minor modifications to the CARLA leaderboard code for the Longest6 benchmark, which are documented [here](./leaderboard). See the [leaderboard/data/longest6](./leaderboard/data/longest6/) folder for a description of Longest6 and how to evaluate on it.

### Pretrained agents
Pre-trained agent files for all 4 methods can be downloaded from [AWS](https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip):

```Shell
mkdir model_ckpt
wget https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip -P model_ckpt
unzip model_ckpt/models_2022.zip -d model_ckpt/
rm model_ckpt/models_2022.zip
```

### Running an agent
To evaluate a model, we first launch a CARLA server:

```Shell
./CarlaUE4.sh --world-port=2000 -opengl
```

Once the CARLA server is running, evaluate an agent with the script:
```Shell
./leaderboard/scripts/local_evaluation.sh <carla root> <working directory of this repo (*/transfuser/)>
```

By editing the arguments in `local_evaluation.sh`, we can benchmark performance on the Longest6 routes. You can evaluate both privileged agents (such as [autopilot.py]) and sensor-based models. To evaluate the sensor-based models use [submission_agent.py](./team_code_transfuser/submission_agent.py) as the `TEAM_AGENT` and point to the folder you downloaded the model weights into for the `TEAM_CONFIG`. The code is automatically configured to use the correct method based on the args.txt file in the model folder.

You can look at qualitative examples of the expected driving behavior of TransFuser on the Longest6 routes [here](https://www.youtube.com/watch?v=DZS-U3-iV0s&list=PL6LvknlY2HlQG3YQ2nMIx7WcnyzgK9meO).


## Credits

This repository hevealy depends on the following repos:

- End-to-end driving with Semantic Depth Cloud [Github](https://github.com/oskarnatan/end-to-end-driving)

- DATA from TransFuser PAMI 2022  [paper](https://arxiv.org/abs/2205.15997)

- Transfuser CVPR 2021  [Github](https://github.com/autonomousvision/transfuser/tree/cvpr2021).

- CvT netowrk CVPR 2021 [Github](https://github.com/microsoft/CvT), [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)

- TCP NeurIPS 2022  [Github](https://github.com/OpenPerceptionX/TCP)



<!-- ### Building docker image

Add the following paths to your ```~/.bashrc```
```
export CARLA_ROOT=<path_to_carla_root>
export SCENARIO_RUNNER_ROOT=<path_to_scenario_runner_in_this_repo>
export LEADERBOARD_ROOT=<path_to_leaderboard_in_this_repo>
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

Edit the contents of ```leaderboard/scripts/Dockerfile.master``` to specify the required dependencies, agent code and model checkpoints. Add all the required information in the area delimited by the tags ```BEGINNING OF USER COMMANDS``` and ```END OF USER COMMANDS```. The current Dockerfile works for all the models in this repository.

Specify a name for the docker image in ```leaderboard/scripts/make_docker.sh``` and run:
```
leaderboard/scripts/make_docker.sh
```

Refer to the Transfuser example for the directory structure and where to include the code and checkpoints.

### Testing the docker image locally

Spin up a CARLA server:
```
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -world-port=2000 -opengl
```

Run the docker container:  
Docker 19:  
```
docker run -it --rm --net=host --gpus '"device=0"' -e PORT=2000 <docker_image> ./leaderboard/scripts/run_evaluation.sh
```
If the docker container doesn't start properly, add another environment variable ```SDL_AUDIODRIVER=dsp```.

### Submitting docker image to the leaderboard

Register on [AlphaDriver](https://app.alphadrive.ai/), create a team and apply to the CARLA Leaderboard.

Install AlphaDrive cli:
```
curl http://dist.alphadrive.ai/install-ubuntu.sh | sh -
```

Login to alphadrive and submit the docker image:
```
alpha login
alpha benchmark:submit --split <2/3> <docker_image>
```
Use ```split 2``` for MAP track and ```split 3``` for SENSORS track. -->
