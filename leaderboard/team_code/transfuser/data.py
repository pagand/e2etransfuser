import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import cv2

class CARLA_Data(Dataset):

    def __init__(self, root, config):
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.ignore_sides = config.ignore_sides
        self.ignore_rear = config.ignore_rear

        self.input_resolution = config.input_resolution
        self.scale = config.scale

        self.lidar = []
        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []
        
        for sub_root in tqdm(root, file=sys.stdout):
            preload_file = os.path.join(sub_root, 'rg_lidar_diag_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'.npy')

            # dump to npy if no preload
            if not os.path.exists(preload_file):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_rear = []
                preload_lidar = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []

                # list sub-directories in root 
                root_files = os.listdir(sub_root)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
                for route in routes:
                    route_dir = os.path.join(sub_root, route)
                    print(route_dir)
                    # subtract final frames (pred_len) since there are no future waypoints
                    # first frame of sequence not used
                    
                    num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
                    
                    for seq in range(num_seq):
                        fronts = []
                        lefts = []
                        rights = []
                        rears = []
                        lidars = []
                        xs = []
                        ys = []
                        thetas = []

                        # read files sequentially (past and current frames)
                        for i in range(self.seq_len):
                            # images
                            filename = f"{str(seq*self.seq_len+1+i).zfill(4)}.png"
                            fronts.append(route_dir+"/rgb_front/"+filename)
                            lefts.append(route_dir+"/rgb_left/"+filename)
                            rights.append(route_dir+"/rgb_right/"+filename)
                            rears.append(route_dir+"/rgb_rear/"+filename)

                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        # get control value of final frame in sequence
                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        # read files sequentially (future frames)
                        for i in range(self.seq_len, self.seq_len + self.pred_len):
                            # point cloud
                            lidars.append(route_dir + f"/lidar/{str(seq*self.seq_len+1+i).zfill(4)}.npy")
                            
                            # position
                            with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])

                            # fix for theta=nan in some measurements
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_lidar.append(lidars)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)

                # dump to npy
                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['lidar'] = preload_lidar
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.lidar += preload_dict.item()['lidar']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['rears'] = []
        data['lidars'] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]
        seq_lidars = self.lidar[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        full_lidar = []
        pos = []
        neg = []
        for i in range(self.seq_len):
            data['fronts'].append(torch.from_numpy(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_sides:
                data['lefts'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_lefts[i]), scale=self.scale, crop=self.input_resolution))))
                data['rights'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rights[i]), scale=self.scale, crop=self.input_resolution))))
            if not self.ignore_rear:
                data['rears'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_rears[i]), scale=self.scale, crop=self.input_resolution))))
            
            lidar_unprocessed = np.load(seq_lidars[i])[...,:3] # lidar: XYZI
            full_lidar.append(lidar_unprocessed)
        
            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.

        ego_x = seq_x[i]
        ego_y = seq_y[i]
        ego_theta = seq_theta[i]

        # future frames
        for i in range(self.seq_len, self.seq_len + self.pred_len):
            lidar_unprocessed = np.load(seq_lidars[i])
            full_lidar.append(lidar_unprocessed)      

        # lidar and waypoint processing to local coordinates
        waypoints = []
        for i in range(self.seq_len + self.pred_len):
            # waypoint is the transformed version of the origin in local coordinates
            # we use 90-theta instead of theta
            # LBC code uses 90+theta, but x is to the right and y is downwards here
            local_waypoint = transform_2d_points(np.zeros((1,3)), 
                np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
            waypoints.append(tuple(local_waypoint[0,:2]))

            # process only past lidar point clouds
            if i < self.seq_len:
                # convert coordinate frame of point cloud
                full_lidar[i][:,1] *= -1 # inverts x, y
                full_lidar[i] = transform_2d_points(full_lidar[i], 
                    np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
                lidar_processed = lidar_to_histogram_features(full_lidar[i], crop=self.input_resolution)
                data['lidars'].append(lidar_processed)

        data['waypoints'] = waypoints

        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = tuple(local_command_point)

        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['command'] = self.command[index]
        data['velocity'] = self.velocity[index]
        
        return data


def lidar_to_histogram_features(lidar, crop=256):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-2*x_meters_max, 2*x_meters_max+1, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[...,:2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.0]
    above = lidar[lidar[...,2]>-2.0]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([below_features, above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features

def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[1] += 1.3
    point = target_point * 8.
    point[1] *= -1
    point[1] = 256 - point[1] 
    point[0] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float) / 255.
    
def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop[0]//2
    start_y = width//2 - crop[1]//2
    cropped_image = image[start_x:start_x+crop[0], start_y:start_y+crop[1]]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image
    
def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out


def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    """
    Convert LiDAR point cloud to camera co-ordinates

    world: Expects the point cloud from CARLA in the CARLA coordinate system: x left, y forward, z up (LiDAR rotated by 90 degree)
    lidar_vis: lidar prjected to BEV
    image_vis: RGB input image to the network
    step: current timestep
    debug: Whether to save the debug images. If false only world is required
    """

    pixels_per_meter = 8
    lidar_width      = 256
    lidar_height     = 256
    lidar_meters_x   = (lidar_width  / pixels_per_meter) / 2 # Divided by two because the LiDAR is in the center of the image
    lidar_meters_y   =  lidar_height / pixels_per_meter

    downscale_factor = 32

    img_width  = 352
    img_height = 160
    fov_width  = 60

    left_camera_rotation  = -60.0
    right_camera_rotation =  60.0

    fov_height = 2.0 * np.arctan((img_height / img_width) * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)

    # Our pixels are squares so focal_x = focal_y
    focal_x = img_width  / (2.0 * np.tan(np.deg2rad(fov_width)  / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))

    cam_z   = 2.3
    lidar_z = 2.5

    # get valid points in 64x64 grid
    world[:, 0] *= -1  # flip x axis, so that the positive direction points towards right. new coordinate system: x right, y forward, z up
    lidar = world[abs(world[:,0])<lidar_meters_x] # 32m to the sides
    lidar = lidar[lidar[:,1]<lidar_meters_y] # 64m to the front
    lidar = lidar[lidar[:,1]>0] # 0m to the back

    # Translate Lidar cloud to the same coordinate system as the cameras (They only differ in height)
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)

    # Make copies because we will rotate the new pointclouds
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)


    lidar_indices = np.arange(0, lidar.shape[0], 1)
    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar[..., 1]
    x = ((focal_x * lidar[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar[..., 2]) / z) + (img_height / 2.0)
    result_center = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
    result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]

    result_center_shifted = result_center
    result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)

    # Rotate the left camera to align with the axis for projection with a pinhole camera model
    theta = np.radians(left_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_left_camera[..., 1]
    x = ((focal_x * lidar_for_left_camera[..., 0]) / z) + (img_width  / 2.0)
    y = ((focal_y * lidar_for_left_camera[..., 2]) / z) + (img_height / 2.0)
    result_left = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
    result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
    result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)

    # Do the same for the right image
    theta = np.radians(right_camera_rotation)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])
    lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T

    # Use a pinhole camera model to project the LiDAR points onto the camera image
    z = lidar_for_right_camera[..., 1]
    x = ((focal_x * lidar_for_right_camera[..., 0]) / z) + (img_width / 2.0)
    y = ((focal_y * lidar_for_right_camera[..., 2]) / z) + (img_height / 2.0)
    result_right = np.stack([x, y, lidar_indices], 1)

    # Remove points that are outside of the image
    result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
    result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]

    # We only use half of the left image, so we cut the unneccessary points
    result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] # Cut of right part, it's not used.
    result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width

    # Combine the three images into one
    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    if(debug == True):
        # Visualize LiDAR hits in image
        vis = np.zeros([img_height, 2 * img_width])
        vis_bev = np.zeros([lidar_height, lidar_width])
        vis_original_image = image_vis[0].detach().cpu().numpy()
        vis_original_image = np.transpose(vis_original_image, (1, 2, 0)) / 255.0
        vis_original_lidar = np.zeros([lidar_height, lidar_width])
        lidar_vis = lidar_vis.detach().cpu().numpy()
        vis_original_lidar[np.greater(lidar_vis[0,0], 0)] = 255
        vis_original_lidar[np.greater(lidar_vis[0,1], 0)] = 255


    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        # Project the LiDAR point to BEV and save index of the BEV image pixel.
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height-1)) * -1

        valid_bev_points.append([bev_x, bev_y])
        # Calculate index in the final image by rounding down
        img_x = int(results_total[i][0])
        # The network input images use a top left coordinate system, we need to convert the bottom left coordinates by inverting the y axis
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])


        if (debug == True):
            vis_original_image[img_y, img_x] = np.array([0.0,1.0,0.0])
            vis_bev[bev_y, bev_x] = 255 #Debug visualization
            vis[img_y, img_x] = 255

    if (debug == True):
        # NOTE add the paths you want the images to land in here before debugging
        from matplotlib import pyplot as plt
        plt.ion()
        plt.imshow(vis_bev)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/bev_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.imshow(vis_original_image)
        plt.savefig(r'/home/hiwi/save folder/Visualizations/2/image_with_lidar_{}.png'.format(step), bbox_inches='tight')
        plt.close()
        plt.ioff()


    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)

    return bev_points, cam_points

