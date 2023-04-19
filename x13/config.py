import os
import random

class GlobalConfig:
    num_worker = 0# for debugging 0
    wandb = False
    gpu_id = '0'
    model = 'April18_cvt_02_solar_debug'
    wandb_name = model 
    logdir = 'log/'+model
    model = 'randomized_low_data' # for wandb

    kind = 'min_cvt' # ['effnet', cvt_effnet', 'cvt_cnn','min_cvt'] # for version1,2 min_cvt change the bottleneck and network arch in this config

#    model = 'speed_cmd(out1cvt)'  # run name
    model = 'x13_control_'  # run name

#    num_worker = 4# for debugging 0
#    gpu_id = '0'
#    wandb = False
#    low_data = True
#    wandb_name = 'x13_small_data'
    #wandb_name = 'randomized_low_data'

     # TODO: correct the forward path in case of change
    kind = 'min_cvt' # ['effnet', cvt_effnet', 'cvt_cnn','min_cvt'] # for version1,2 min_cvt change the bottleneck and network arch in this config

# #    model = 'speed_cmd(out1cvt)'  # run name
 #   model = 'x13_control_'  # run name

 #   model += kind+'_v2'
 #   logdir = 'log/'+model #+'_w1' for 1 weather only


    init_stop_counter = 15
    n_class = 23
    
    batch_size = 2 #20
    total_epoch = 30
    
    low_data = True
    low_data_rate = 0.2

    # MGN parameter
    MGN = True   ## True
    loss_weights = [1, 1, 1, 1, 1, 1, 1]
    lw_alpha = 1.5

    if kind == 'cvt_cnn':
        bottleneck = [350, 695, 350]
    elif kind == 'min_cvt':
        # version 1
        # bottleneck = [342, 687, 332]
        # version 2
        bottleneck = [332, 683, 332]
    else:
        bottleneck = [335, 679, 335]

    n_class = 23
    batch_size = 20 #20
    total_epoch = 20 #30

    random_data_len = int(170740 *0.2) #int(188660 * 0.2 ) 
    cvt_freezed_epoch = 0  # nonzero only for version 1 Min-CVT

    if kind == 'cvt_effnet' or kind == 'effnet':
        # parameters for Effnet
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] 
    elif kind == 'cvt_cnn':
    # parameters for CVT
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        n_fmap_b3 = [[40,32], [64], [192], [96,384], [232,384,1536]] 
    elif kind == 'min_cvt':
        # version 1
        # n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        # n_fmap_b3 = [[32,24], [64], [192], [96,384], [232,384,1536]]
        # # version 2
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,112]] 
        n_fmap_b3 = [[32,24], [64], [192], [96,1536, 384], [232,384,384]]  
    else:
        raise Exception("The kind of architecture is not recognized. choose form these in the config: ['effnet', cvt_effnet', 'cvt_cnn']")
    


    # MGN parameter
    MGN = True
    loss_weights = [1, 1, 1, 1, 1, 1, 0, 1]
    lw_alpha = 1.5

	# for Data
    seq_len = 1 # jumlah input seq
    pred_len = 3 # future waypoints predicted

    # root_dir = '/home/aisl/OSKAR/Transfuser/transfuser_data/14_weathers_full_data'  #14_weathers_full_data OR clear_noon_full_data
    # root_dir = '/localhome/pagand/projects/e2etransfuser/data'  # for the CVPR dataset
    root_dir = '/home/mohammad/Mohammad_ws/autonomous_driving/transfuser/data'  # '/localscratch/mmahdavi/transfuser/data' # for the PAMI dataset
    #root_dir = '/localhome/pagand/projects/e2etransfuser/transfuser_pmlr/data'
    train_data, val_data = [], []

    ## For PMLR dataset'/localscratch/mmahdavi/transfuser/data'
    root_files = os.listdir(root_dir)
    # train_towns = ['Town04']
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10HD']
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
<<<<<<< HEAD
        val_data = random.sample(val_data,int(low_data_rate*len(val_data)))
=======
        train_data = random.sample(train_data,int(0.2*len(train_data)))
        val_data = random.sample(val_data,int(0.2*len(val_data)))
>>>>>>> 77054189e9b2ac64d57adcf18a55c46cd4484453

        # train_data = train_data[:int(0.05*len(train_data))]
        # val_data = val_data[:int(0.1*len(val_data))]

    #buat prediksi expert, test
    test_data = []
    test_weather = 'Run3_ClearNoon' #ClearNoon, ClearSunset, CloudyNoon, CloudySunset, WetNoon, WetSunset, MidRainyNoon, MidRainSunset, WetCloudyNoon, WetCloudySunset, HardRainNoon, HardRainSunset, SoftRainNoon, SoftRainSunset, Run1_ClearNoon, Run2_ClearNoon, Run3_ClearNoon
    test_scenario = 'ADVERSARIAL' #NORMAL ADVERSARIAL
    expert_dir = '/media/aisl/data/XTRANSFUSER/EXPERIMENT_RUN/8T1W/EXPERT/'+test_scenario+'/'+test_weather  #8T1W 8T14W
    for town in val_towns: #test town = val town
        test_data.append(os.path.join(expert_dir, 'Expert_w1')) #Expert Expert_w1

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    # input_resolution = [256,256] # CVPR dataset
    # input_resolution = 160 # PMLR dataset
    input_resolution = [160,768] # PMLR dataset #768
    # input_resolution = [160,160] # PMLR dataset #512
  #  coverage_area = [64,64]
    coverage_area = [64/256*input_resolution[0],64/256*input_resolution[1]]  #64

    # camera intrinsic
    img_width = 352
    img_height = 160
    fov = 2*60
    
    scale = 1 # image pre-processing
    # crop = 256 # image pre-processing # CVPR dataset
    crop = 160 # image pre-processing # CVPR dataset
    lr = 1e-4 # learning rate AdamW
    weight_decay = 1e-3

    # Controller
    #control weights untuk PID dan MLP dari tuningan MGN
    #urutan steer, throttle, brake
    #baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    #LWS: lw_wp lw_str lw_thr lw_brk saat convergence
    lws = [0.999884128570556, 1.00005769729614, 0.999974191188812, 1.00002384185791]
    # : [0.999923467636108, 1.00002813339233, 0.999968945980072, 0.999997556209564]
    # _w1 : [0.99992048740387, 1.00002562999725, 0.999992907047272, 1.00002014636993]
    # _t2 : [0.999884128570556, 1.00005769729614, 0.999974191188812, 1.00002384185791]
    # _t2w1 : [0.9998899102211, 1.0000467300415, 0.999997615814209, 1.00002729892731]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2]), lws[0]/(lws[0]+lws[3])] #str, thrt, brk
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1], 1-cw_pid[2]] #str, thrt, brk


    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.2 #minimum throttle

    #ORDER DALAM RGB!!!!!!!!
    SEG_CLASSES = {
        'colors'        :[[0, 0, 0], [70, 70, 70], [100, 40, 40], [55, 90, 80], [220, 20, 60],  
                            [153, 153, 153], [157, 234, 50], [128, 64, 128], [244, 35, 232], [107, 142, 35], 
                            [0, 0, 142], [102, 102, 156], [220, 220, 0], [70, 130, 180], [81, 0, 81],
                            [150, 100, 100], [230, 150, 140], [180, 165, 180], [250, 170, 30], [110, 190, 160],
                            [170, 120, 50], [45, 60, 150], [145, 170, 100]], 
        'classes'       : ['None', 'Building', 'Fences', 'Other', 'Pedestrian',
                            'Pole', 'RoadLines', 'Road', 'Sidewalk', 'Vegetation',
                            'Vehicle', 'Wall', 'TrafficSign', 'Sky', 'Ground',
                            'Bridge', 'RailTrack', 'GuardRail', 'TrafficLight', 'Static',
                            'Dynamic', 'Water', 'Terrain']
    }

    if kind == 'cvt_effnet' or kind == 'effnet' or kind == 'rest':
        # parameters for Effnet
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] 
    elif kind == 'cvt_cnn':
    # parameters for CVT
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        n_fmap_b3 = [[40,32], [64], [192], [96,384], [232,384,1536]] 
    elif kind == 'min_cvt':
        # version 1
        # n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,1280]] 
        # n_fmap_b3 = [[32,24], [64], [192], [96,384], [232,384,1536]]
        # # version 2
        n_fmap_b1 = [[32,16], [24], [40], [80,112], [192,320,112]] 
        n_fmap_b3 = [[32,24], [64], [192], [96,1536, 384], [232,384,384]] 
    else:
        raise Exception("The kind of architecture is not recognized. choose form these in the config: ['effnet', cvt_effnet', 'cvt_cnn']")

    ## fusion settings
    attn = True
    fusion_embed_dim_q = n_fmap_b3[3][-1] #n_fmap_b3[4][-1]
    fusion_embed_dim_kv = n_fmap_b1[3][-1]
    fusion_depth = 4 #1
    fusion_num_heads = 8 #1
    fusion_mlp_ratio = 4
    fusion_qkv = True
    fusion_drop_rate = 0
    fusion_attn_drop_rate = 0
    fusion_dpr = [0,0,0,0] # [0.1,0.2,0.3,0.4]

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
