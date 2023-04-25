import os
import random

class GlobalConfig:
    """ base architecture configurations """
    num_worker = 4# for debugging 0
    gpu_id = '0'
    wandb = False
    low_data = True
    wandb_name = 'baselines'

    kind = 'baseline' 

    model = 'transfuser_'
    model += kind
    logdir = 'log/'+model #+'_w1' for 1 weather only

    total_epoch = 40
    batch_size = 50
    val_cycle = 1

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

    root_dir = '/home/oskar/OSKAR/Transfuser/transfuser_data/14_weathers_full_data'
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10HD']
    val_towns = ['Town05']
    # train_towns = ['Town00']
    # val_towns = ['Town00']

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
        train_data = random.sample(train_data,int(0.2*len(train_data)))
        val_data = random.sample(val_data,int(0.2*len(val_data)))

    # visualizing transformer attention maps
    # TODO what is viz_data
    # viz_root = '/mnt/qb/geiger/kchitta31/data_06_21'
    # viz_towns = ['Town05_tiny']
    # viz_data = []
    # for town in viz_towns:
    #     viz_data.append(os.path.join(viz_root, town))

    ignore_sides = False # don't consider side cameras
    ignore_rear = True # don't consider rear cameras
    n_views = 1 # no. of camera views

    input_resolution = (160, 704)
    camera_width = 960
    camera_height = 480
    img_width_cut = 320
    
    img_resolution = (160,704)

    # camera intrinsic
    img_width = 320 # 352
    img_height = 160
    fov = 2*60
    
    scale = 1 # image pre-processing
    crop = 160 # image pre-processing

    lr = 1e-4 # learning rate

    # Conv Encoder
    vert_anchors = int(input_resolution[1]/32) #8
    horz_anchors = int(input_resolution[0]/32) #8
    anchors = vert_anchors * horz_anchors

    ##
    augment = True
    inv_augment_prob = 0.1 # Probablity that data augmentation is applied is 1.0 - inv_augment_prob
    aug_max_rotation = 20 # degree
    debug = False # If true the model in and outputs will be visualized and saved into Os variable Save_Path
    sync_batch_norm = False # If this is true we convert the batch norms, to synced bach norms.
    train_debug_save_freq = 50 # At which interval to save debug files to disk during training

    bb_confidence_threshold = 0.3 # Confidence of a bounding box that is needed for the detection to be accepted
    bounding_box_divisor = 2.0 # The height and width of the bounding box value was changed by this factor during data collection. Fix that for future datasets and remove
    draw_brake_threshhold = 0.5 # If the brake value is higher than this threshhold, the bb will be drawn with the brake color during visualization


    # GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 4
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 20 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 20 # buffer size

    default_speed = 4.0 # Speed used when creeping
    clip_throttle = 0.75 # Maximum throttle allowed by the controller
    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    ## added
    use_target_point_image = True
    gru_concat_target_point = True
    use_point_pillars = False
    
    # Conv Encoder
    img_vert_anchors = 5
    img_horz_anchors = 20 + 2
    lidar_vert_anchors = 8
    lidar_horz_anchors = 8
    img_seq_len = 1 
    lidar_seq_len = 1

    lidar_resolution_width  = 256 # Width of the LiDAR grid that the point cloud is voxelized into.
    lidar_resolution_height = 256 # Height of the LiDAR grid that the point cloud is voxelized into.
    pixels_per_meter = 8.0 # How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
    lidar_pos = [1.3,0.0,2.5] # x, y, z mounting position of the LiDAR
    lidar_rot = [0.0, 0.0, -90.0] # Roll Pitch Yaw of LiDAR in degree

    gpt_linear_layer_init_mean = 0.0 # Mean of the normal distribution with which the linear layers in the GPT are initialized
    gpt_linear_layer_init_std  = 0.02 # Std  of the normal distribution with which the linear layers in the GPT are initialized
    gpt_layer_norm_init_weight = 1.0 # Initial weight of the layer norms in the gpt.
    perception_output_features = 512 # Number of features outputted by the perception branch.
    bev_features_chanels = 64 # Number of channels for the BEV feature pyramid
    bev_upsample_factor = 2

    deconv_channel_num_1 = 128 # Number of channels at the first deconvolution layer
    deconv_channel_num_2 = 64 # Number of channels at the second deconvolution layer
    deconv_channel_num_3 = 32 # Number of channels at the third deconvolution layer

    deconv_scale_factor_1 = 8 # Scale factor, of how much the grid size will be interpolated after the first layer
    deconv_scale_factor_2 = 4 # Scale factor, of how much the grid size will be interpolated after the second layer

    gps_buffer_max_len = 100 # Number of past gps measurements that we track.
    carla_frame_rate = 1.0 / 20.0 # CARLA frame rate in milliseconds
    carla_fps = 20 # Simulator Frames per second
    iou_treshold_nms = 0.2  # Iou threshold used for Non Maximum suppression on the Bounding Box predictions for the ensembles
    steer_damping = 0.5 # Damping factor by which the steering will be multiplied when braking
    route_planner_min_distance = 7.5
    route_planner_max_distance = 50.0
    action_repeat = 2 # Number of times we repeat the networks action. It's 2 because the LiDAR operates at half the frame rate of the simulation
    stuck_threshold = 1100/action_repeat # Number of frames after which the creep controller starts triggering. Divided by
    creep_duration = 30 / action_repeat # Number of frames we will creep forward

    # Size of the safety box
    safety_box_z_min = -2.0
    safety_box_z_max = -1.05

    safety_box_y_min = -3.0
    safety_box_y_max = 0.0

    safety_box_x_min = -1.066
    safety_box_x_max = 1.066

    ego_extent_x = 2.4508416652679443 # Half the length of the ego car in x direction
    ego_extent_y = 1.0641621351242065 # Half the length of the ego car in x direction
    ego_extent_z = 0.7553732395172119 # Half the length of the ego car in x direction




    # Optimization
    multitask = True # whether to use segmentation + depth losses
    ls_seg   = 1.0
    ls_depth = 10.0
    gru_hidden_size = 64
    num_class = 7
    classes = {
        0: [0, 0, 0],  # unlabeled
        1: [0, 0, 255],  # vehicle
        2: [128, 64, 128],  # road
        3: [255, 0, 0],  # red light
        4: [0, 255, 0],  # pedestrian
        5: [157, 234, 50],  # road line
        6: [255, 255, 255],  # sidewalk
    }
    #Color format BGR
    classes_list = [
        [0, 0, 0],  # unlabeled
        [255, 0, 0],  # vehicle
        [128, 64, 128],  # road
        [0, 0, 255],  # red light
        [0, 255, 0],  # pedestrian
        [50, 234, 157],  # road line
        [255, 255, 255],  # sidewalk
    ]
    converter = [
        0,  # unlabeled
        0,  # building
        0,  # fence
        0,  # other
        4,  # pedestrian
        0,  # pole
        5,  # road line
        2,  # road
        6,  # sidewalk
        0,  # vegetation
        1,  # vehicle
        0,  # wall
        0,  # traffic sign
        0,  # sky
        0,  # ground
        0,  # bridge
        0,  # rail track
        0,  # guard rail
        0,  # traffic light
        0,  # static
        0,  # dynamic
        0,  # water
        0,  # terrain
        3,  # red light
        3,  # yellow light
        0,  # green light
        0,  # stop sign
        5,  # stop line marking
    ]
    
    # CenterNet parameters
    num_dir_bins = 12
    fp16_enabled = False
    center_net_bias_init_with_prob = 0.1
    center_net_normal_init_std = 0.001
    top_k_center_keypoints = 100
    center_net_max_pooling_kernel = 3  
    channel = 64
        
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
