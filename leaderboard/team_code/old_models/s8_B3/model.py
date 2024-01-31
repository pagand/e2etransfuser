from collections import deque
import sys
import numpy as np
from torch import torch, cat, add, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms




#FUNGSI INISIALISASI WEIGHTS MODEL
#baca https://pytorch.org/docs/stable/nn.init.html
#kaiming he
def kaiming_w_init(layer, nonlinearity='relu'):
    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
    layer.bias.data.fill_(0.01)

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
        #weights initialization
        kaiming_w_init(self.conv)
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, final=False): #up, 
        super(ConvBlock, self).__init__()
        #conv block
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=1)
            self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=1)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
 
    def forward(self, x):
        #convolutional block
        y = self.conv_block0(x)
        y = self.conv_block1(y)
        return y


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control



class s8_B3(nn.Module): #
    #default input channel adalah 3 untuk RGB, 2 untuk DVS, 1 untuk LiDAR
    def __init__(self, config, device):#n_fmap, n_class=[23,10], n_wp=5, in_channel_dim=[3,2], spatial_dim=[240, 320], gpu_device=None): 
        super(s8_B3, self).__init__()
        self.config = config
        n_fmap_b3 = [[40,24], [32], [48], [96,136], [232,384,1536]] #lihat underdevelopment/efficientnet.py
        # n_fmap_b0 = [[32,16], [24], [40], [80,112], [192,320,1280]]
        # n_fmap_b4 = [[48,24], [32], [56], [112,160], [272,448,1792]]
        # n_fmap_b2 = [[32,16], [24], [48], [88,120], [208,352,1408]]
        self.gpu_device = device
        self.flatten = nn.Flatten()
        #------------------------------------------------------------------------------------------------
        #RGB, jika inputnya sequence, maka jumlah input channel juga harus menyesuaikan
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.RGB_encoder = models.efficientnet_b3(pretrained=True) #efficientnet_b4
        self.RGB_encoder.classifier = nn.Sequential() #cara paling gampang untuk menghilangkan fc layer yang tidak diperlukan
        self.RGB_encoder.avgpool = nn.Sequential()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        #SS
        self.conv3_ss_f = ConvBlock(channel=[n_fmap_b3[4][-1]+n_fmap_b3[3][-1], n_fmap_b3[3][-1]])#, up=True)
        self.conv2_ss_f = ConvBlock(channel=[n_fmap_b3[3][-1]+n_fmap_b3[2][-1], n_fmap_b3[2][-1]])#, up=True)
        self.conv1_ss_f = ConvBlock(channel=[n_fmap_b3[2][-1]+n_fmap_b3[1][-1], n_fmap_b3[1][-1]])#, up=True)
        self.conv0_ss_f = ConvBlock(channel=[n_fmap_b3[1][-1]+n_fmap_b3[0][-1], n_fmap_b3[0][0]])#, up=True)
        self.final_ss_f = ConvBlock(channel=[n_fmap_b3[0][0], config.n_class], final=True)#, up=False)
        #------------------------------------------------------------------------------------------------
        #waypoint predition, input size selalu 4 karena concat dari x,y wp dan x,y rooute next
        self.n_wp = config.pred_len
        #jumlahkan kedua bottleneck
        self.necks_net = nn.Sequential( #inputnya sum dari 2 bottleneck
            ConvBNRelu(channelx=[n_fmap_b3[4][-1], n_fmap_b3[4][1]], kernelx=1, paddingx=0, stridex=1),
            # STNet(ch=[n_fmap_b3[4][1], n_fmap_b3[3][-1], n_fmap_b3[2][-1]]),
            ConvBNRelu(channelx=[n_fmap_b3[4][1], n_fmap_b3[4][1]], kernelx=2, paddingx=0, stridex=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_fmap_b3[4][1], n_fmap_b3[4][0]),
            nn.BatchNorm1d(n_fmap_b3[4][0]),
            nn.ReLU()
        )
        #embedding hidden state dari vehicle location, next route, and velocity
        self.gru_nxr = nn.GRUCell(input_size=2, hidden_size=n_fmap_b3[4][0]) #gru untuk next route
        self.gru_vel = nn.GRUCell(input_size=1, hidden_size=n_fmap_b3[4][0]) #gru untuk current velocity

        #wp predictor
        self.gru_wp = nn.GRUCell(input_size=2, hidden_size=n_fmap_b3[4][0])#nn.ModuleList([nn.GRUCell(input_size=2, hidden_size=n_fmap_b3[2][-1]) for _ in range(config.pred_len)])
        self.pred_dwp = nn.Linear(n_fmap_b3[4][0], 2)

        #controller
        #MLP Controller
        self.controller = nn.Sequential(
            nn.Linear(n_fmap_b3[4][0], n_fmap_b3[3][-1]),
            nn.ReLU(),
            nn.Linear(n_fmap_b3[3][-1], n_fmap_b3[3][0]),
            nn.ReLU(),
            nn.Linear(n_fmap_b3[3][0], 3),
            nn.ReLU()
        )
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        

    def forward(self, rgb_f, next_route, velo_in):#, gt_ss): depth_f
        #------------------------------------------------------------------------------------------------
        #bagian downsampling
        RGB_features_sum = 0
        for i in range(self.config.seq_len): #loop semua input dalam buffer
            in_rgb = self.rgb_normalizer(rgb_f[i])
            RGB_features0 = self.RGB_encoder.features[0](in_rgb)
            RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
            RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
            RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
            RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
            RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
            RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
            RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
            RGB_features8 = self.RGB_encoder.features[8](RGB_features7)
            RGB_features_sum += RGB_features8
        #bagian upsampling
        ss_f_3 = self.conv3_ss_f(cat([self.up(RGB_features8), RGB_features5], dim=1))
        ss_f_2 = self.conv2_ss_f(cat([self.up(ss_f_3), RGB_features3], dim=1))
        ss_f_1 = self.conv1_ss_f(cat([self.up(ss_f_2), RGB_features2], dim=1))
        ss_f_0 = self.conv0_ss_f(cat([self.up(ss_f_1), RGB_features1], dim=1))
        ss_f = self.final_ss_f(self.up(ss_f_0))

        #------------------------------------------------------------------------------------------------
        #waypoint prediction
        #get hidden state dari gabungan kedua bottleneck
        hid_states = self.necks_net(RGB_features_sum) #hid_state0)# cat([RGB_features_sum, SC_features8], dim=1)
        #hid_states dari next route dan velocity
        hid_state_nxr = self.gru_nxr(next_route, hid_states)
        hid_state_vel = self.gru_vel(torch.reshape(velo_in, (velo_in.shape[0], 1)), hid_states)

        #jumlahkan hidden state
        hx = hid_states+hid_state_nxr+hid_state_vel
        # initial input car location ke GRU, selalu buat batch size x 2 (0,0) (xy)
        xy = torch.zeros(size=(hx.shape[0], 2)).float().to(self.gpu_device)
        #predict delta wp
        out_wp = list()
        for _ in range(self.n_wp):
            hx = self.gru_wp(xy, hx)
            d_xy = self.pred_dwp(hx) 
            xy = xy + d_xy
            out_wp.append(xy)
        pred_wp = torch.stack(out_wp, dim=1)

        #control decoder
        control_pred = self.controller(hx) #cat([hid_states, hid_state_nxr, hid_state_vel], dim=1)
        steer = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        brake = control_pred[:,2]

        return ss_f, pred_wp, steer, throttle, brake#, top_view_sc

    def mlp_pid_control(self, waypoints, velocity, mlp_steer, mlp_throttle, mlp_brake, ctrl_opt):
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        pid_steer = self.turn_controller.step(angle)
        pid_steer = np.clip(pid_steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        pid_throttle = self.speed_controller.step(delta)
        pid_throttle = np.clip(pid_throttle, 0.0, self.config.max_throttle)
        pid_brake = 0.0
        #constrain dari brake flag
        # if desired_speed < self.config.brake_speed or (speed/desired_speed) > self.config.brake_ratio:
        #     pid_throttle = 0.0
        #     pid_brake = 1.0
        # else: #jika tidak maka ya jalan seperti biasanya
        #     pid_brake = 0.0

        #final decision
        if ctrl_opt == "one_of":
            #opsi 1: jika salah satu controller aktif, maka vehicle jalan. vehicle berhenti jika kedua controller non aktif
            steer = np.clip(self.config.cw_pid[0]*pid_steer + self.config.cw_mlp[0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle >= self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                steer = pid_steer
                throttle = pid_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle >= self.config.min_act_thrt):
                pid_brake = 1.0
                steer = mlp_steer
                throttle = mlp_throttle
            elif (pid_throttle < self.config.min_act_thrt) and (mlp_throttle < self.config.min_act_thrt):
                throttle = 0.0
                pid_brake = 1.0
                brake = 1.0#np.clip(self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
        elif ctrl_opt == "both_must":
            #opsi 2: vehicle jalan jika dan hanya jika kedua controller aktif. jika salah satu saja non aktif, maka vehicle berhenti
            steer = np.clip(self.config.cw_pid[0]*pid_steer + self.config.cw_mlp[0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle < self.config.min_act_thrt) or (mlp_throttle < self.config.min_act_thrt):
                throttle = 0.0
                pid_brake = 1.0
                brake = 1.0#np.clip(self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
        elif ctrl_opt == "pid_only":
            #opsi 3: PID only
            steer = pid_steer
            throttle = pid_throttle
            brake = 0.0
            #MLP full off
            mlp_steer = 0.0
            mlp_throttle = 0.0
            mlp_brake = 0.0
            if pid_throttle < self.config.min_act_thrt:
                throttle = 0.0
                pid_brake = 1.0
                brake = 1.0#pid_brake
        elif ctrl_opt == "mlp_only":
            #opsi 4: MLP only
            steer = mlp_steer
            throttle = mlp_throttle
            brake = 0.0
            #PID full off
            pid_steer = 0.0
            pid_throttle = 0.0
            pid_brake = 0.0
            if mlp_throttle < self.config.min_act_thrt:
                throttle = 0.0
                brake = 1.0#mlp_brake
        elif ctrl_opt == "pid_def":
            #opsi 5: PID def
            steer = pid_steer
            throttle = pid_throttle
            brake = 0.0
            if desired_speed < self.config.brake_speed or (speed/desired_speed) > self.config.brake_ratio:
                throttle = 0.0
                pid_brake = 1.0
                brake = pid_brake
        else:
            sys.exit("ERROR, FALSE CONTROL OPTION")

        metadata = {
            'control_option': ctrl_opt,
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1]), float(self.config.cw_pid[2])],
            'pid_steer': float(pid_steer),
            'pid_throttle': float(pid_throttle),
            'pid_brake': float(pid_brake),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1]), float(self.config.cw_mlp[2])],
            'mlp_steer': float(mlp_steer),
            'mlp_throttle': float(mlp_throttle),
            'mlp_brake': float(mlp_brake),
            # 'wp_4': tuple(waypoints[3].astype(np.float64)), #tambahan
            'wp_3': tuple(waypoints[2].astype(np.float64)), #tambahan
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
            'car_pos': None, #akan direplace di fungsi agent
            'next_point': None, #akan direplace di fungsi agent
        }
        return steer, throttle, brake, metadata

