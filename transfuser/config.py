import os
import random

class GlobalConfig:
    """ base architecture configurations """
    num_worker = 4# for debugging 0
    gpu_id = '0'
    wandb = True
    low_data = True
    wandb_name = 'baseline'

    kind = 'baseline' 

    model = 'transfuser_'
    model += kind
    logdir = 'log/'+model #+'_w1' for 1 weather only

    total_epoch = 40
    batch_size = 50
    val_cycle = 2

	# Data
    seq_len = 1 # input timesteps
    pred_len = 3 # future waypoints predicted

    # root_dir = '/home/aisl/OSKAR/Transfuser/transfuser_data/14_weathers_full_data'  #14_weathers_full_data clear_noon_full_data
    # train_towns = ['Town01']#, 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    # val_towns = ['Town02']
    # # train_towns = ['Town00']
    # # val_towns = ['Town00']
    # train_data, val_data = [], []
    # for town in train_towns:
    #     if not (town == 'Town07' or town == 'Town10'):
    #         train_data.append(os.path.join(root_dir, town+'_long'))
    #     train_data.append(os.path.join(root_dir, town+'_short'))
    #     train_data.append(os.path.join(root_dir, town+'_tiny'))
    #     # train_data.append(os.path.join(root_dir, town+'_x'))
    # for town in val_towns:
    #     # val_data.append(os.path.join(root_dir, town+'_long'))
    #     val_data.append(os.path.join(root_dir, town+'_short'))
    #     val_data.append(os.path.join(root_dir, town+'_tiny'))
    #     # val_data.append(os.path.join(root_dir, town+'_x'))

    # PMLR data
    root_dir = '/localhome/pagand/projects/e2etransfuser/transfuser_pmlr/data'  # for the PMLR dataset

    train_data, val_data = [], []

    ## For PMLR dataset
    root_files = os.listdir(root_dir)
    # train_towns = ['Town04']
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    val_towns = ['Town05'] # 'Town05'

    for dir in root_files:
        scn_files = os.listdir(os.path.join(root_dir,dir))
        for routes in scn_files:
            for t in routes.split("_"):
                if t[0] != 'T':
                    continue
                if t in train_towns:
                    train_data.append(os.path.join(root_dir,dir, routes))
                    break
                elif t in val_towns:
                    val_data.append(os.path.join(root_dir,dir, routes))
                    break
                else:
                    break
    if low_data:
        random.seed(0)
        train_data = random.sample(train_data,int(0.02*len(train_data)))
        val_data = random.sample(val_data,int(0.1*len(val_data)))

    # visualizing transformer attention maps
    # TODO what is viz_data
    # viz_root = '/mnt/qb/geiger/kchitta31/data_06_21'
    # viz_towns = ['Town05_tiny']
    # viz_data = []
    # for town in viz_towns:
    #     viz_data.append(os.path.join(viz_root, town))

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views



    # input_resolution = 256
    input_resolution = [160,768] # PMLR dataset #768


    scale = 1 # image pre-processing
    crop = 160 #256 # image pre-processing

    lr = 1e-4 # learning rate

    # Conv Encoder
    vert_anchors = int(input_resolution[1]/32) #8
    horz_anchors = int(input_resolution[0]/32) #8
    anchors = vert_anchors * horz_anchors

	# GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.1 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    # TODO: what is test_data
    # #buat prediksi expert, test
    # test_data = []
    # test_weather = 'ClearNoon' #ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset, MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset, HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset, Run1_ClearNoon, Run2_ClearNoon, Run3_ClearNoon
    # test_scenario = 'NORMAL' #NORMAL ADVERSARIAL
    # expert_dir = '/media/aisl/data/XTRANSFUSER/EXPERIMENT_RUN/8T14W/EXPERT/'+test_scenario+'/'+test_weather  #8T1W 8T14W
    # for town in val_towns: #test town = val town
    #     test_data.append(os.path.join(expert_dir, 'Expert')) #Expert Expert_w1
    
    # #untuk yang langsung dari dataset
    # # test_data = ['/media/aisl/data/XTRANSFUSER/my_dataset/clear_noon_full_data/Town05_long']
    # # test_data = ['/media/aisl/HDD/OSKAR/TRANSFUSER/EXPERIMENT_RUN/8T1W/EXPERT/NORMAL/Run2_ClearNoon/Expert_w1']

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
