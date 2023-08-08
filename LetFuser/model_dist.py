from collections import deque, OrderedDict
import sys
import numpy as np
from torch import torch, cat, add, nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CvtModel #, AutoImageProcessor


# can be ignored
import matplotlib.pyplot as plt
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
import time
import os
import cv2
from torchvision.transforms.functional import rotate
# can be ignored




def kaiming_init_layer(layer):
    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class ConvBNRelu(nn.Module):
    def __init__(self, channelx, stridex=1, kernelx=3, paddingx=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(channelx[0], channelx[1], kernel_size=kernelx, stride=stridex, padding=paddingx, padding_mode='zeros')
        self.bn = nn.BatchNorm2d(channelx[1])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        y = self.relu(x)
        return y

class ConvBlock(nn.Module):
    def __init__(self, channel, final=False, stridex=1): #up, 
        super(ConvBlock, self).__init__()
        if final:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[0]], stridex=stridex)
            self.conv_block1 = nn.Sequential(
            nn.Conv2d(channel[0], channel[1], kernel_size=1),
            nn.Sigmoid()
            )
        else:
            self.conv_block0 = ConvBNRelu(channelx=[channel[0], channel[1]], stridex=stridex)
            self.conv_block1 = ConvBNRelu(channelx=[channel[1], channel[1]], stridex=1)
        self.conv_block0.apply(kaiming_init)
        self.conv_block1.apply(kaiming_init)
 
    def forward(self, x):
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
    
class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self,
                 dim_q,
                 num_heads=8,
                 attn_drop=0.,
                 ):
        super().__init__()
        self.dim = dim_q
        self.attn_drop = attn_drop
        # self.q_lin = nn.Linear(self.dim,self.dim)
        # self.kv_lin = nn.Linear(self.dim,2*self.dim)
        # self.fusion = nn.MultiheadAttention(dim_q,num_heads,attn_drop)
        self.q_lin = nn.Linear(dim_q,dim_q)
        self.kv_lin = nn.Linear(dim_q,2*dim_q)
        self.fusion = nn.MultiheadAttention(dim_q,num_heads,attn_drop,batch_first=True)

    def forward(self,q,kv):
        q_end = self.q_lin(q.unsqueeze(1))
        kv = self.kv_lin(kv.unsqueeze(1))
        k_end,v_end = kv.chunk(2,dim=-1)
        fused_data = self.fusion(q_end,k_end,v_end)
        return fused_data[0].squeeze(1)
    

class Attention_2D(nn.Module):
    def __init__(self,
                 dim_q,
                 dim_kv,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False,
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_q
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_q ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_q, dim_q, kernel_size, padding_q,
            1, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_kv, dim_q, kernel_size, padding_kv,
            1, method
        )
        self.conv_proj_v = self._build_projection(
            dim_kv, dim_q, kernel_size, padding_kv,
            1, method
        )

        self.proj_q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_q, dim_q, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_q, dim_q)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=1 #dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_out)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, y, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h*w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(y)
        else:
            k = rearrange(y, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(y)
        else:
            v = rearrange(y, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, y, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, y, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Fusion_Block(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.with_cls_token = False

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention_2D(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
        )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim_in)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_in,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, features, h, w):
        res = features

        x = self.norm1(features)
        attn = self.attn(x, x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x
    
    
class letfuser(nn.Module): #
    def __init__(self, config, device):
        super(letfuser, self).__init__()
        self.config = config
        self.gpu_device = device
        #------------------------------------------------------------------------------------------------
        # CVT and effnet
        # self.pre = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
        if config.kind == "min_cvt":
            self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")
            self.conv1_down = ConvBNRelu(channelx=[3, config.n_fmap_b3[0][-1]],stridex=2)
            # # version2 does not require conv2_down
            # self.conv2_down = ConvBNRelu(channelx=[config.n_fmap_b3[3][-1], config.n_fmap_b3[4][-1]],stridex=2, kernelx = 1, paddingx =0)
            self.conv1_down.apply(kaiming_init)
            # self.conv2_down.apply(kaiming_init)
            
            
        elif config.kind == "cvt_cnn":
            #CVT and conv
            self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")
            self.conv1_down =  ConvBlock(channel=[3, config.n_fmap_b3[0][-1]],stridex=2)
            self.conv2_down =  ConvBlock(channel=[config.n_fmap_b3[3][-1], config.n_fmap_b3[4][-1]],stridex=2)

        elif config.kind == "cvt_effnet":
            self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")
            self.avgpool = nn.AvgPool2d(2, stride=2)

        elif config.kind == "cvt_effnet" or config.kind == "effnet":
            #RGB
            self.RGB_encoder = models.efficientnet_b3(pretrained=True) #efficientnet_b4
            self.RGB_encoder.classifier = nn.Sequential()
            self.RGB_encoder.avgpool = nn.Sequential()  
        
        #RGB
        self.rgb_normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        #SS
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        
        if config.kind == "min_cvt":
            self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_b3[3][-1], config.n_fmap_b3[3][-1]])
        else:
            self.conv3_ss_f = ConvBlock(channel=[config.n_fmap_b3[4][-1]+config.n_fmap_b3[3][-1], config.n_fmap_b3[3][-1]])
        
        self.conv2_ss_f = ConvBlock(channel=[config.n_fmap_b3[3][-1]+config.n_fmap_b3[2][-1], config.n_fmap_b3[2][-1]])
        self.conv1_ss_f = ConvBlock(channel=[config.n_fmap_b3[2][-1]+config.n_fmap_b3[1][-1], config.n_fmap_b3[1][-1]])
        self.conv0_ss_f = ConvBlock(channel=[config.n_fmap_b3[1][-1]+config.n_fmap_b3[0][-1], config.n_fmap_b3[0][0]])
        self.final_ss_f = ConvBlock(channel=[config.n_fmap_b3[0][0], config.n_class], final=True)
        #------------------------------------------------------------------------------------------------
        #red light and stop sign predictor
        # option 1 (for min_cvt version 1/2 and Effnet)
        self.tls_predictor = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][-1], 1),
            nn.Sigmoid()
        )
        #        self.tls_biasing = nn.Linear(1, config.n_fmap_b3[4][0])
        self.tls_biasing_bypass = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]),
            nn.Sigmoid()
        )
        # self.tls_biasing_bypass = nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0])
        #nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0])
        #------------------------------------------------------------------------------------------------
        #SDC
        self.cover_area = config.coverage_area
        self.n_class = config.n_class
        self.h, self.w = config.input_resolution[0], config.input_resolution[1]
	
	    #fx = self.config.img_width / (2 * np.tan(self.config.fov * np.pi / 360))
        #fy = self.config.img_height / (2 * np.tan(fovh * np.pi / 360))

        # fx = 160# 160 (for fov 86 deg, 300 image size)
        #self.x_matrix = torch.vstack([torch.arange(-self.w/2, self.w/2)]*self.h) / fx

        fovh = np.rad2deg(2.0 * np.arctan((self.config.img_height / self.config.img_width) * np.tan(0.5 * np.radians(self.config.fov))))
        # self.fx = self.config.img_width / (2 * np.tan(self.config.fov * np.pi / 360))
        fy = self.config.img_height / (2 * np.tan(fovh * np.pi / 360))

        self.fx = 160  # 160 
        self.x_matrix = torch.vstack([torch.arange(-self.w/2, self.w/2)]*self.h) / self.fx
        self.x_matrix = self.x_matrix.to(device)
        #SC
        self.SC_encoder = models.efficientnet_b1(pretrained=False) 
        self.SC_encoder.features[0][0] = nn.Conv2d(config.n_class, config.n_fmap_b1[0][0], kernel_size=3, stride=2, padding=1, bias=False) 
        self.SC_encoder.classifier = nn.Sequential() 
        self.SC_encoder.avgpool = nn.Sequential()
        self.SC_encoder.apply(kaiming_init)
        #------------------------------------------------------------------------------------------------
        #feature fusion
        if config.attn:
            embed_dim_q = self.config.fusion_embed_dim_q
            embed_dim_kv = self.config.fusion_embed_dim_kv
            depth = self.config.fusion_depth
            num_heads = self.config.fusion_num_heads
            mlp_ratio = self.config.fusion_mlp_ratio
            qkv_bias = self.config.fusion_qkv
            drop_rate = self.config.fusion_drop_rate
            attn_drop_rate = self.config.fusion_attn_drop_rate
            dpr = self.config.fusion_dpr
            act_layer=nn.GELU
            norm_layer =nn.LayerNorm

            self.attn_neck = nn.Sequential( #inputnya dari 2 bottleneck
            nn.Conv2d(config.fusion_embed_dim_q+config.fusion_embed_dim_kv, config.n_fmap_b3[4][1], kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][1], config.n_fmap_b3[4][0])
            )
        else:
            self.necks_net = nn.Sequential( #inputnya dari 2 bottleneck
                nn.Conv2d(config.n_fmap_b3[4][-1]+config.n_fmap_b1[4][-1], config.n_fmap_b3[4][1], kernel_size=1, stride=1, padding=0),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(config.n_fmap_b3[4][1], config.n_fmap_b3[4][0])
            )
        #------------------------------------------------------------------------------------------------
        #Speed predictor
        # self.speed_head = nn.Sequential(
        #                     nn.AdaptiveAvgPool2d(1),
        #                     nn.Flatten(),
		# 					nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[3][0]),
		# 					nn.ReLU(inplace=True),
		# 					# nn.Linear(256, 256),
		# 					# nn.Dropout2d(p=0.5),
		# 					# nn.ReLU(inplace=True),
		# 					nn.Linear(config.n_fmap_b3[3][0], 1),
		# 				)
        
        
        self.speed_head = nn.Sequential(
                nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[3][0]),
                nn.ReLU(inplace=True),
                # nn.Linear(256, 256),
                # nn.Dropout2d(p=0.5),
                # nn.ReLU(inplace=True),
                nn.Linear(config.n_fmap_b3[3][0], 1),
            )
        # comment 1

        self.fuse_BN = nn.BatchNorm2d(config.n_fmap_b3[-1][-1]+config.n_fmap_b1[-1][-1])
        self.measurements = nn.Sequential(
							nn.Linear(1+2+6, config.n_fmap_b1[-1][-1]),
							nn.ReLU(inplace=True),
							nn.Linear(config.n_fmap_b1[-1][-1], config.n_fmap_b3[3][0]),
							nn.ReLU(inplace=True),
						)
        self.gru_control = nn.GRUCell(input_size=3+2, hidden_size=config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]) #control version2 +0  ,  control v4 +2
        self.pred_control = nn.Sequential(
            # nn.Linear(2*config.n_fmap_b3[4][0], 3), #v1
            # nn.Sigmoid()
            nn.Linear(2*config.n_fmap_b3[4][0]+2*config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]), #v2
            nn.ReLU(inplace=True), #CHANGED!
            nn.Linear(config.n_fmap_b3[3][-1], 3),
            #nn.Tanh()  # v5 Sigmoid  #CHANGED!

            # nn.Linear(config.n_fmap_b3[4][0], 3), #v3
            # nn.Sigmoid()
            )
        #------------------------------------------------------------------------------------------------
        #wp predictor, input size 5 karena concat dari xy, next route xy, dan velocity
        # self.gru = nn.GRUCell(input_size=5+6, hidden_size=config.n_fmap_b3[4][0])
        # self.gru = nn.GRUCell(input_size=5, hidden_size=config.n_fmap_b3[4][0])
        self.gru = nn.GRUCell(input_size=5-1, hidden_size=config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0])
        self.pred_dwp = nn.Linear(2*config.n_fmap_b3[4][0]+2*config.n_fmap_b3[3][0], 2) #control v4
        #PID Controller
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)
        #------------------------------------------------------------------------------------------------
        #controller
        #MLP Controller
        # self.controller = nn.Sequential(
        #     nn.Linear(config.n_fmap_b3[3][0], config.n_fmap_b3[3][0]//2),
        #     nn.Linear(config.n_fmap_b3[3][0]//2, 3),
        #     nn.ReLU()
        # )
        self.controller = nn.Sequential(
            nn.Linear(config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]),
            nn.Linear(config.n_fmap_b3[3][-1], 3),
            nn.ReLU()
        )
        if config.attn:
            blocks = []
            for j in range(depth):
                blocks.append(
                Fusion_Block(
                    dim_in=embed_dim_q+embed_dim_kv,
                    dim_out=embed_dim_q+embed_dim_kv,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                )
            self.blocks = nn.ModuleList(blocks)
            self.input_buffer = {'depth': deque()}


        #------------------------------------------------------------------------------------------------
        # for distilation
        self.D_tls_biasing_bypass = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]),
            nn.Sigmoid()
        )
        self.D_tls_biasing_bypass2 = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]),
            nn.Sigmoid()
        )
        self.D_tls_biasing_bypass3 = nn.Sequential( 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.n_fmap_b3[4][-1], config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]),
            nn.Sigmoid()
        )
        self.D_gru = nn.GRUCell(input_size=5-1, hidden_size=config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0])
        self.D_pred_dwp = nn.Linear(2*config.n_fmap_b3[4][0]+2*config.n_fmap_b3[3][0], 2)


        self.D_controller = nn.Sequential(
            nn.Linear(config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]),
            nn.Linear(config.n_fmap_b3[3][-1], 1),
            nn.ReLU()
        )
        self.D_controller3 = nn.Sequential(
            nn.Linear(config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]),
            nn.Linear(config.n_fmap_b3[3][-1], 1),
            nn.ReLU()
        )
        self.D_gru_control = nn.GRUCell(input_size=1+2, hidden_size=config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]) #control version2 +0  ,  control v4 +2
        self.D_gru_control3 = nn.GRUCell(input_size=1+2, hidden_size=config.n_fmap_b3[4][0]+config.n_fmap_b3[3][0]) #control version2 +0  ,  control v4 +2
        self.D_pred_control = nn.Sequential(
            nn.Linear(2*config.n_fmap_b3[4][0]+2*config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]), #v2
            nn.ReLU(inplace=True),
            nn.Linear(config.n_fmap_b3[3][-1], 1),
            )
        self.D_pred_control3 = nn.Sequential(
            nn.Linear(2*config.n_fmap_b3[4][0]+2*config.n_fmap_b3[3][0], config.n_fmap_b3[3][-1]), #v2
            nn.ReLU(inplace=True),
            nn.Linear(config.n_fmap_b3[3][-1], 1),
            )

    def forward(self, rgb_f, depth_f, next_route, velo_in, gt_command ):#, gt_ss, gt_redl:
        #------------------------------------------------------------------------------------------------
        # CVT and conv (approach2) and Min CVT
        in_rgb = self.rgb_normalizer(rgb_f) #[i]
        out = self.cvt(in_rgb, output_hidden_states=True)
        RGB_features1 = self.conv1_down(in_rgb)
        RGB_features2 = out[2][0]
        RGB_features3 = out[2][1]
        RGB_features5 = out[2][2]
        # # version2 does not require conv2_down
        # RGB_features8 = self.conv2_down(RGB_features5) # version 1
        RGB_features8 = RGB_features5 # version 2
        # TODO: for Min CVT change upsampling
        # TODO: for min_CVT version 2 change hx to use SC_features5
        # TODO: fer version 2, comment conv2_down in init

        # # CVT and effnet (approach1)
        # # inputs = self.pre(rgb_f, return_tensors="pt").to(self.gpu_device)
        # # out = self.cvt(**inputs, output_hidden_states=True)
        # embed_dim = [24, 32, 48, 136]
        # in_rgb = self.rgb_normalizer(rgb_f) #[i]
        # out = self.cvt(in_rgb, output_hidden_states=True)
        # RGB_features1 = self.RGB_encoder.features[0](in_rgb)[:,:embed_dim[0],:,:]
        # RGB_features2 = out[2][0][:,:embed_dim[1],:,:]
        # RGB_features3 = out[2][1][:,:embed_dim[2],:,:]
        # RGB_features5 = out[2][2][:,:embed_dim[3],:,:]
        # RGB_features9 = self.RGB_encoder.features[8](out[2][2])
        # RGB_features8 = self.avgpool(RGB_features9)
        # ss_f_3 = self.conv3_ss_f(cat([RGB_features9, RGB_features5], dim=1))
        # # TODO: Comment next conv0_ss_f
        # # TODO: change self.necks_net for version 2 and the SC_features after 5

        # only Effnet
        #in_rgb = self.rgb_normalizer(rgb_f) #[i]
        #RGB_features0 = self.RGB_encoder.features[0](in_rgb)
        #RGB_features1 = self.RGB_encoder.features[1](RGB_features0)
        #RGB_features2 = self.RGB_encoder.features[2](RGB_features1)
        #RGB_features3 = self.RGB_encoder.features[3](RGB_features2)
        #RGB_features4 = self.RGB_encoder.features[4](RGB_features3)
        #RGB_features5 = self.RGB_encoder.features[5](RGB_features4)
        #RGB_features6 = self.RGB_encoder.features[6](RGB_features5)
        #RGB_features7 = self.RGB_encoder.features[7](RGB_features6)
        #RGB_features8 = self.RGB_encoder.features[8](RGB_features7)

        # bagian upsampling
        # ss_f = self.conv3_ss_f(cat([self.up(RGB_features8), RGB_features5], dim=1))
        # # only for Min CVT (both versions)
        ss_f = self.conv3_ss_f(RGB_features5)

        ss_f = self.conv2_ss_f(cat([self.up(ss_f), RGB_features3], dim=1))
        ss_f = self.conv1_ss_f(cat([self.up(ss_f), RGB_features2], dim=1))
        ss_f = self.conv0_ss_f(cat([self.up(ss_f), RGB_features1], dim=1))
        ss_f = self.final_ss_f(self.up(ss_f))

        bs,ly,wi,hi = ss_f.shape
        #------------------------------------------------------------------------------------------------
        #create a semantic cloud
        if False: #self.show:
            big_top_view = torch.zeros((bs,ly,2*wi,2*hi)).cuda()
            for i in range(3):
                if i==0:
                    width = 224 # 224
                    depth_f_p = depth_f[:,:,:,:width]
                    ss_f_p = gt_ss[:,:,:,:width]
                    rot = 130 #60 # 43.3
                    height_coverage = 120
                    width_coverage = 300
                elif i==1:
                    width = 224 # 224
                    depth_f_p = depth_f[:,:,:,-width:]
                    ss_f_p = gt_ss[:,:,:,-width:]
                    rot = -65 #-60 # -43.3
                    height_coverage = 120
                    width_coverage = 300
                elif i==2:
                    width = 320 # 320
                    depth_f_p = depth_f[:,:,:,224:768-224]
                    ss_f_p = gt_ss[:,:,:,224:768-224]
                    rot = 0
                    height_coverage = 160
                    width_coverage = 320

                big_top_view = self.gen_top_view_sc_show(big_top_view, depth_f_p, ss_f_p, rot, width, hi,height_coverage,width_coverage) #  ss_f  ,rgb_f

            big_top_view = big_top_view[:,:,0:wi,768-160:768+160]
            self.save2(gt_ss,big_top_view)
        
        big_top_view = torch.zeros((bs,ly,2*wi,hi)).cuda()
        for i in range(3):
            if i==0:
                width = 224 # 224
                rot = 130 #60 # 43.3
                height_coverage = 120
                width_coverage = 300
                big_top_view = self.gen_top_view_sc(big_top_view, depth_f[:,:,:,:width], ss_f[:,:,:,:width], rot, width, hi, height_coverage,width_coverage)
            elif i==1:
                width = 224 # 224
                rot = -65 #-60 # -43.3
                height_coverage = 120
                width_coverage = 300
                big_top_view = self.gen_top_view_sc(big_top_view, depth_f[:,:,:,-width:], ss_f[:,:,:,-width:], rot, width, hi, height_coverage,width_coverage)
            elif i==2:
                width = 320 # 320
                rot = 0
                height_coverage = 160
                width_coverage = 320
                big_top_view = self.gen_top_view_sc(big_top_view, depth_f[:,:,:,224:hi-224], ss_f[:,:,:,224:hi-224], rot, width, hi,height_coverage,width_coverage)

        top_view_sc = big_top_view[:,:,:wi,:]

        #downsampling section
        #------------------------------------------------------------------------------------------------
        #buat semantic cloud
        #top_view_sc = self.gen_top_view_sc(depth_f, ss_f ) # ss_f gt_ss ,rgb_f
        #bagan downsampling
        SC_features0 = self.SC_encoder.features[0](top_view_sc)
        SC_features1 = self.SC_encoder.features[1](SC_features0)
        SC_features2 = self.SC_encoder.features[2](SC_features1)
        SC_features3 = self.SC_encoder.features[3](SC_features2)
        SC_features4 = self.SC_encoder.features[4](SC_features3)
        SC_features5 = self.SC_encoder.features[5](SC_features4)
        # for min-cvt version 2 should be commented
        # SC_features6 = self.SC_encoder.features[6](SC_features5)
        # SC_features7 = self.SC_encoder.features[7](SC_features6)
        # SC_features8 = self.SC_encoder.features[8](SC_features7)

        #------------------------------------------------------------------------------------------------
        #red light and stop sign detection
        redl_stops = self.tls_predictor(RGB_features8)
        red_light = redl_stops[:,0] #gt_redl
        tls_bias = self.tls_biasing_bypass(RGB_features8)
        bs,_,H,W = RGB_features8.shape
        #------------------------------------------------------------------------------------------------
        #Speed prediction
        speed = self.speed_head(out[1].squeeze(-2)) # RGB_features8
        #------------------------------------------------------------------------------------------------
        #red light and stop sign detection
        # stop_sign = redl_stops[:,1]  # we don't have stop sign
        stop_sign = torch.zeros_like(red_light)
        #tls_bias = self.tls_biasing(redl_stops)   #   gt_redl.unsqueeze(-1)
        #------------------------------------------------------------------------------------------------
        #waypoint prediction: get hidden state dari gabungan kedua bottleneck

        # hx = self.necks_net(cat([RGB_features8, SC_features8], dim=1)) #RGB_features_sum+SC_features8 cat([RGB_features_sum, SC_features8], dim=1)
        # # for min_CVT version 2

        # hx = self.necks_net(cat([RGB_features8, SC_features5], dim=1))
        
        # No attention TODO 1 if not config.atten
        # measurement_feature = self.measurements(torch.cat([next_route, velo_in.unsqueeze(-1), F.one_hot((gt_command-1).to(torch.int64).long(), num_classes=6)], dim=1))
        # fuse = self.fuse_BN(torch.cat([RGB_features8, SC_features5], dim=1))
        # hx = self.necks_net(fuse)
        # hx = torch.cat([hx, measurement_feature], dim=1) 
        # fuse = hx.clone()#NEW

        # With attention TODO 1 if config.atten
        measurement_feature = self.measurements(torch.cat([next_route, velo_in.unsqueeze(-1), F.one_hot((gt_command-1).to(torch.int64).long(), num_classes=6)], dim=1))
        fuse = self.fuse_BN(torch.cat([RGB_features8, SC_features5], dim=1))
        features_cat = rearrange(fuse , 'b c h w-> b (h w) c')
        for i, blk in enumerate(self.blocks):
            x = blk(features_cat, H, W)
        x = rearrange(x , 'b (h w) c-> b c h w', h=H,w=W)
        hx = self.attn_neck(x)
        hx = torch.cat([hx, measurement_feature], dim=1) 
        fuse = hx.clone()#NEW
        

        ## 
        xy = torch.zeros(size=(hx.shape[0], 2)).float().to(self.gpu_device)
        # predict delta wp
        out_wp = list()

        # distilation single task
        D_tls_bias = self.D_tls_biasing_bypass(RGB_features8)
        D_xy = torch.zeros(size=(hx.shape[0], 2)).float().to(self.gpu_device)
        D_out_wp = list()
        D_hx = hx.clone()#NEW
        D_hx2 = hx.clone()
        D_hx3 = hx.clone()
        D_tls_bias2 = self.D_tls_biasing_bypass2(RGB_features8)
        D_tls_bias3 = self.D_tls_biasing_bypass3(RGB_features8)

        for _ in range(self.config.pred_len):
            # ins = torch.cat([xy, next_route, velo_in.unsqueeze(-1), F.one_hot((gt_command-1).to(torch.int64).long(), num_classes=6)], dim=1) # x
            # ins = torch.cat([xy, next_route, velo_in.unsqueeze(-1)], dim=1) # x
            ins = torch.cat([xy, next_route], dim=1) # x
            hx = self.gru(ins, hx)
            # d_xy = self.pred_dwp(hx+tls_bias) #why adding??
            d_xy = self.pred_dwp(torch.cat([hx,tls_bias], dim=1)) #control v4
            xy = xy + d_xy
            out_wp.append(xy)

            # distilation single task
            D_hx = self.D_gru(ins, D_hx)
            D_d_xy = self.D_pred_dwp(torch.cat([D_hx,D_tls_bias], dim=1)) #control v4
            D_xy = D_xy + D_d_xy
            D_out_wp.append(D_xy)

        pred_wp = torch.stack(out_wp, dim=1)
        D_pred_wp = torch.stack(D_out_wp, dim=1)

        # computing error for distilation single task (wp)
        D_feature_loss = torch.sum((D_hx-hx)*(D_hx-hx))+ torch.sum((D_tls_bias-tls_bias)*( D_tls_bias-tls_bias)) 
        #------------------------------------------------------------------------------------------------
        
        # distilation single task (steer)
        D_control_pred = self.D_controller(D_hx2+D_tls_bias2)
        out_control = list()
        for _ in range(1): # TODO 2   if self.config.augment_control_data then range(self.config.pred_len)  o.w. range(1)
            ins = torch.cat([D_control_pred, next_route], dim=1) # control v4
            D_hx2 = self.D_gru_control(ins, D_hx2) # control v5
            d_control = self.D_pred_control(torch.cat([D_hx2,D_tls_bias2], dim=1)) # control v2
            D_control_pred = D_control_pred + d_control # control v2/3/4
            out_control.append(D_control_pred)
        pred_control = torch.stack(out_control, dim=1)
        D_steer = pred_control[:,:,0]* 2 - 1.

        # distilation single task (brake)
        D_control_pred = self.D_controller3(D_hx3+D_tls_bias3)
        out_control = list()
        
        for _ in range(1): # TODO 2   if self.config.augment_control_data then range(self.config.pred_len)  o.w. range(1)
            ins = torch.cat([D_control_pred, next_route], dim=1) # control v4
            D_hx3 = self.D_gru_control3(ins, D_hx3) # control v5
            d_control = self.D_pred_control3(torch.cat([D_hx3,D_tls_bias3], dim=1)) # control v2
            D_control_pred = D_control_pred + d_control # control v2/3/4
            out_control.append(D_control_pred)
        pred_control = torch.stack(out_control, dim=1)
        D_brake = pred_control[:,:,0]
        



        #control decoder
        control_pred = self.controller(hx+tls_bias)
        
        # TODO 2 comment  if self.config.augment_control_data
        # out_control = list()
        # for _ in range(self.config.pred_len):
        #     ins = torch.cat([control_pred, next_route], dim=1) # control v4
        #     hx = self.gru_control(ins, hx) # control v5
        #     d_control = self.pred_control(torch.cat([hx,tls_bias], dim=1)) # control v2
        #     control_pred = control_pred + d_control # control v2/3/4
        #     out_control.append(control_pred)
        # pred_control = torch.stack(out_control, dim=1)
        # steer = pred_control[:,:,0]* 2 - 1.
        # throttle = pred_control[:,:,1] * self.config.max_throttle
        # brake = pred_control[:,:,2] #brake: hard 1.0 or no 0.0

        
        # TODO  2 comment  if not self.config.augment_control_data 
        ins = torch.cat([control_pred, next_route], dim=1) # control v4
        # ins = control_pred# control v2
        hx = self.gru_control(ins, fuse) # control v2/3/4
        d_control = self.pred_control(torch.cat([hx,tls_bias], dim=1)) # control v2
        # d_control = self.pred_control(hx+tls_bias)  # making add (#v3)
        control_pred = control_pred + d_control # control v2/3/4
        steer = control_pred[:,0] * 2 - 1. # convert from [0,1] to [-1,1]
        throttle = control_pred[:,1] * self.config.max_throttle
        brake = control_pred[:,2] #brake: hard 1.0 or no 0.0



        # computing error for distilation single task (steer)
        D_feature_loss += torch.sum((D_hx2-hx)*(D_hx2-hx))\
                                            + torch.sum((D_hx3-hx)*(D_hx3-hx))\
                                            + torch.sum((D_tls_bias2-tls_bias)*( D_tls_bias2-tls_bias)) \
                                            + torch.sum((D_tls_bias3-tls_bias)*( D_tls_bias3-tls_bias))
        

        return ss_f, pred_wp, steer, throttle, brake, red_light, stop_sign, top_view_sc, speed, D_pred_wp, D_steer,  D_brake, D_feature_loss # redl_stops[:,0] , top_view_sc       

    def scale_and_crop_image_cv(self, image, scale=1, crop=256):
        upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
        cropped_image = np.transpose(cropped_im, (2,0,1))
        return cropped_image
    
    def rgb_to_depth(self, de_gt):
        de_gt = de_gt.transpose(1, 2, 0)
        arrayd = de_gt.astype(np.float32)
        normalized_depth = np.dot(arrayd, [65536.0, 256.0, 1.0]) # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        depthx = normalized_depth/16777215.0  # (256.0 * 256.0 * 256.0 - 1.0) --> rangenya 0 - 1
        result = np.expand_dims(depthx, axis=0)
        return result

    def swap_RGB2BGR(self,matrix):
        red = matrix[:,:,0].copy()
        blue = matrix[:,:,2].copy()
        matrix[:,:,0] = blue
        matrix[:,:,2] = red
        return matrix

    def get_wp_nxr_frame(self):
        frame_dim = self.config.crop - 1 #array mulai dari 0
        area = self.config.coverage_area

        point_xy = []
	    #proses wp
        for i in range(1, self.config.pred_len+1):
            x_point = int((frame_dim/2) + (self.control_metadata['wp_'+str(i)][0]*(frame_dim/2)/area[1]))
            y_point = int(frame_dim - (self.control_metadata['wp_'+str(i)][1]*frame_dim/area[0]))
            xy_arr = np.clip(np.array([x_point, y_point]), 0, frame_dim) #constrain
            point_xy.append(xy_arr)
	
	    #proses juga untuk next route
	    # - + y point kebalikan dari WP, karena asumsinya agent mendekati next route point, dari negatif menuju 0
        x_point = int((frame_dim/2) + (self.control_metadata['next_point'][0]*(frame_dim/2)/area[1]))
        y_point = int(frame_dim + (self.control_metadata['next_point'][1]*frame_dim/area[0]))
        xy_arr = np.clip(np.array([x_point, y_point]), 0, frame_dim) #constrain
        point_xy.append(xy_arr)
        return point_xy
		
    def save2(self, ss, sc):
        frame = 0
        ss = ss.cpu().detach().numpy()
        sc = sc.cpu().detach().numpy()

        #buat array untuk nyimpan out gambar
        imgx = np.zeros((ss.shape[2], ss.shape[3], 3))
        imgx2 = np.zeros((sc.shape[2], sc.shape[3], 3))
        #ambil tensor output segmentationnya
        pred_seg = ss[0]
        pred_sc = sc[0]
        inx = np.argmax(pred_seg, axis=0)
        inx2 = np.argmax(pred_sc, axis=0)
        for cmap in self.config.SEG_CLASSES['colors']:
            cmap_id = self.config.SEG_CLASSES['colors'].index(cmap)
            imgx[np.where(inx == cmap_id)] = cmap
            imgx2[np.where(inx2 == cmap_id)] = cmap
	# Image.fromarray(imgx).save(self.save_path / 'segmentation' / ('%06d.png' % frame))
	# Image.fromarray(imgx2).save(self.save_path / 'semantic_cloud' / ('%06d.png' % frame))
	
	#GANTI ORDER BGR KE RGB, SWAP!
        imgx = self.swap_RGB2BGR(imgx)
        imgx2 = self.swap_RGB2BGR(imgx2)

        cv2.imwrite('/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/train_1%06d.png' % frame, imgx) #cetak predicted segmentation
        cv2.imwrite('/home/mohammad/Mohammad_ws/autonomous_driving/e2etransfuser/train_2%06d.png' % frame, imgx2) #cetak predicted segmentation

    def gen_top_view_sc_show(self, big_top_view, depth, semseg, rot, im_width, im_height,height_coverage,width_coverage):
        #proses awal
        self.x_matrix2 = torch.vstack([torch.arange(-im_width//2, im_width//2)]*self.h) / self.fx
        self.x_matrix2 = self.x_matrix2.to('cuda')

        depth_in = depth * 1000.0 #normalisasi ke 1 - 1000
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*im_width)] for n in range(depth.shape[0])])).to(self.gpu_device)
        coverage_area = [64/256*height_coverage,64/256*width_coverage] 

        #normalize to frames
        cloud_data_x = torch.round(((depth_in * self.x_matrix2) + (coverage_area[1]/2)) * (im_width-1) / coverage_area[1]).ravel()
        cloud_data_z = torch.round((depth_in * -(self.h-1) / coverage_area[0]) + (self.h-1)).ravel()

        #look for index interests
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= im_width-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        #stack n x z cls and plot
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW

        bs, ly, wi, hi = top_view_sc.shape
        big_top_view[:,:,0:1*wi,im_height-hi//2:im_height+hi//2] = torch.where(top_view_sc != 0, top_view_sc, big_top_view[:,:,0:1*wi,im_height-hi//2:im_height+hi//2])
        if rot != 0:
            big_top_view = rotate(big_top_view,rot)

        self.save2(semseg,big_top_view)

        return big_top_view
    
    def gen_top_view_sc(self, big_top_view, depth, semseg, rot, im_width, im_height, height_coverage, width_coverage):
        #proses awal
        self.x_matrix2 = torch.vstack([torch.arange(-im_width//2, im_width//2)]*self.h) / self.fx
        self.x_matrix2 = self.x_matrix2.to('cuda')

        depth_in = depth * 1000.0 #normalisasi ke 1 - 1000
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*im_width)] for n in range(depth.shape[0])])).to(self.gpu_device)
        coverage_area = [64/256*height_coverage,64/256*width_coverage] 

        #normalize to frames
        cloud_data_x = torch.round(((depth_in * self.x_matrix2) + (coverage_area[1]/2)) * (im_width-1) / coverage_area[1]).ravel()
        cloud_data_z = torch.round((depth_in * -(self.h-1) / coverage_area[0]) + (self.h-1)).ravel()

        #look for index interests
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= im_width-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

        #stack n x z cls and plot
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
        top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW

        bs, ly, wi, hi = top_view_sc.shape
        big_top_view[:,:,0:1*wi,(im_height-hi)//2:(im_height+hi)//2] = torch.where(top_view_sc != 0, top_view_sc, big_top_view[:,:,0:1*wi,(im_height-hi)//2:(im_height+hi)//2])
        if rot != 0:
            big_top_view = rotate(big_top_view,rot)

        return big_top_view
  
    def gen_top_view_sc_old(self, depth, semseg): #,rgb_f
        #proses awal
        depth_in = depth * 1000.0 #normalize to 1 - 1000
        _, label_img = torch.max(semseg, dim=1) #pada axis C
        cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(self.h*self.w)] for n in range(depth.shape[0])])).to(self.gpu_device)

        #normalize to frame
        cloud_data_x = torch.round(((depth_in * self.x_matrix) + (self.cover_area[1]/2)) * (self.w-1) / self.cover_area[1]).ravel()
        cloud_data_z = torch.round((depth_in * -(self.h-1) / self.cover_area[0]) + (self.h-1)).ravel()

        #find the interest index
        bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= self.w-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= self.h-1, cloud_data_z >= 0))
        idx_xz = bool_xz.nonzero().squeeze() #remove axis with size=1, so no need to add ".item()" later

        #stack n x z cls dan plot
        coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
        coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor must be long so that it can be used as an index

        top_view_sc = torch.zeros_like(semseg) #this is faster because automatically the size, data type, and device are the same as those of the input (semseg)
        top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #axis format from NCHW

        return top_view_sc

    def mlp_pid_control(self, waypoints, velocity, mlp_steer, mlp_throttle, mlp_brake, redl, ctrl_opt="one_of"):
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        red_light = True if redl.data.cpu().numpy() > 0.5 else False

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
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
        elif ctrl_opt == "both_must":
            #opsi 2: vehicle jalan jika dan hanya jika kedua controller aktif. jika salah satu saja non aktif, maka vehicle berhenti
            steer = np.clip(self.config.cw_pid[0]*pid_steer + self.config.cw_mlp[0]*mlp_steer, -1.0, 1.0)
            throttle = np.clip(self.config.cw_pid[1]*pid_throttle + self.config.cw_mlp[1]*mlp_throttle, 0.0, self.config.max_throttle)
            brake = 0.0
            if (pid_throttle < self.config.min_act_thrt) or (mlp_throttle < self.config.min_act_thrt):
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = np.clip(self.config.cw_pid[2]*pid_brake + self.config.cw_mlp[2]*mlp_brake, 0.0, 1.0)
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
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                pid_brake = 1.0
                brake = pid_brake
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
                # steer = 0.0 #dinetralkan
                throttle = 0.0
                brake = mlp_brake
        else:
            sys.exit("ERROR, FALSE CONTROL OPTION")

        metadata = {
            'control_option': ctrl_opt,
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'red_light': float(red_light),
            'cw_pid': [float(self.config.cw_pid[0]), float(self.config.cw_pid[1]), float(self.config.cw_pid[2])],
            'pid_steer': float(pid_steer),
            'pid_throttle': float(pid_throttle),
            'pid_brake': float(pid_brake),
            'cw_mlp': [float(self.config.cw_mlp[0]), float(self.config.cw_mlp[1]), float(self.config.cw_mlp[2])],
            'mlp_steer': float(mlp_steer),
            'mlp_throttle': float(mlp_throttle),
            'mlp_brake': float(mlp_brake),
            'wp_3': tuple(waypoints[2].astype(np.float64)), 
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
