from .networks import ImplicitNet, RenderingNet, OffsetNet
from .density import LaplaceDensity, AbsDensity
from .real_nvp import SimpleNVP
from .projection_layer import get_projection_layer
import numpy as np
import torch
import torch.nn as nn
import os
from pytorch3d import ops

class V2A(nn.Module):
    def __init__(self, opt, keyframe_path, gender, num_training_frames, num_cams):
        super().__init__()

        # Foreground networks
        self.implicit_network = ImplicitNet(opt.implicit_network)
        self.implicit_network_obj = ImplicitNet(opt.implicit_network_obj)
        self.rendering_network = RenderingNet(opt.rendering_network)
        self.d_out = opt.implicit_network.d_out
        self.d_out_obj = opt.implicit_network_obj.d_out
        self.num_semantic = self.d_out
        self.sigmoid = opt.implicit_network.sigmoid
        self.pool = nn.MaxPool1d(self.d_out_obj)
        self.use_deformed_verts = opt.implicit_network.use_romp_verts
        self.pred_skinning_weight = opt.implicit_network.pred_skinning_weight
        self.deformer_type = opt.deformer.type
        self.dist_type = opt.deformer.dist_type
        self.use_keyframe = opt.deformer.use_keyframe
        self.weightnet_use_cond = opt.deformer.network.cond
        self.n_kps = opt.implicit_network.kps
        self.skel_type = opt.implicit_network.skel_type
        self.softmax_type = opt.implicit_network.softmax_type
        self.use_broyden = opt.deformer.use_broyden
        self.debug = opt.debug
        self.debug_img_id = opt.debug_id
        self.offset_net = None   
        self.global_step = 0     

        # Skinning weight prediction networks
        self.skinning_net = ImplicitNet(opt.deformer.network) 
        self.skinning_net_obj = ImplicitNet(opt.deformer.network) 

        if opt.deformer.network.offsetnet == True:
            self.offset_net =  OffsetNet(opt.offsetnet) 

        # Background networks
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(num_training_frames, opt.bg_rendering_network.dim_frame_encoding)
        self.cam_latent_encoder = nn.Embedding(num_cams, opt.bg_rendering_network.dim_frame_encoding) 
        
        self.density = LaplaceDensity(**opt.density)
        self.obj_density = LaplaceDensity(**opt.obj_density)        
        self.bg_density = AbsDensity()

        print("randomly initializing implicitnet weights..")

        key_frame_info = np.load(keyframe_path, allow_pickle=True).item()
        self.jnts_v_cano = torch.Tensor(key_frame_info['keyframe_joints']).unsqueeze(0).cuda()      

        if self.deformer_type == 'inn':
            self.deformer = SimpleNVP(n_layers=2,
                                  feature_dims=24*3 + 3, ###### 3* no of joints + 1,
                                  cano_skel=self.jnts_v_cano[:,:24,:],
                                  hidden_size=512,
                                  projection=get_projection_layer(proj_dims=256, type="simple"),
                                  checkpoint=False,
                                  normalize=True,
                                  explicit_affine=True,
                                  skinning_net=self.skinning_net,
                                  offsetnet=self.offset_net,
                                  dist_type=self.dist_type,
                                  use_cond=self.weightnet_use_cond,
                                  n_kps=24,
                                  skel_type='non_rigid',
                                  softmax_type=self.softmax_type
                                  )
            self.deformer_obj = SimpleNVP(n_layers=2,
                                  feature_dims=24*3 + 3, ###### 3* no of joints + 1,
                                  cano_skel=self.jnts_v_cano[:,24:,:],
                                  hidden_size=512,
                                  projection=get_projection_layer(proj_dims=256, type="simple"),
                                  checkpoint=False,
                                  normalize=True,
                                  explicit_affine=True,
                                  skinning_net=self.skinning_net_obj,
                                  offsetnet=self.offset_net,
                                  dist_type=self.dist_type,
                                  use_cond=self.weightnet_use_cond,
                                  n_kps=24,
                                  skel_type='obj',
                                  softmax_type=self.softmax_type
                                  ) 
    
    def get_sem_label(self, pts, control_points, sigma=0.05):

        #### get distance of points from nearest joints in human
        human_skel = control_points[:,:24,:]
        distance_batch, _, _ = ops.knn_points(pts.unsqueeze(0), human_skel.reshape(1,24,3),
                                                                    K=1, return_nn=True)  
        
        human_dist = torch.exp(-torch.square(distance_batch)/sigma**2)

        #### get distance of points from nearest joints in object
        obj_skel = control_points[:,24:,:]
        distance_batch_obj, _, _ = ops.knn_points(pts.unsqueeze(0), obj_skel.reshape(1,24,3),
                                                                    K=1, return_nn=True)  
        obj_dist = torch.exp(-torch.square(distance_batch_obj)/sigma**2)
        bg_dist = 1.0 - torch.clamp((human_dist+obj_dist), max=1.0)

        sem_indctr = torch.cat([bg_dist, human_dist, obj_dist], -1)

        return sem_indctr
