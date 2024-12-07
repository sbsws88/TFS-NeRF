import torch.nn as nn
import torch
import numpy as np
from .embedders import get_embedder

def dense_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer

class OffsetNet(nn.Module):
    def __init__(self, opt, cano_cond=None):
        super().__init__()

        self.dim_pose_embed = 8
        dims = [opt.d_in] + list(
            opt.dims) + [opt.d_out]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in 
        self.cond_dim = opt.kps * opt.d_in
        self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)      

        self.input_layers = nn.ModuleList()
        in_features = self.dim_pose_embed + dims[0]
        for i in range(1,self.num_layers-1):
            self.input_layers.append(dense_layer(in_features, dims[i]))
            if len(self.skip_in)>0:
                if i % self.skip_in == 0 and i > 0:
                    in_features = dims[i] + opt.d_in
            else:
                in_features = dims[i]
        hidden_features = in_features
        self.sigma_layer = dense_layer(hidden_features, dims[-1])   
        self.offset_act = torch.nn.ReLU() 

    def forward(self, x, cond):
        """Query the sigma (and the features) of the points in canonical space.

        :params x: [..., input_dim]
        :return
            raw_sigma: [..., 1]
            raw_feat: Optional [..., D]
        """
        input_cond = self.lin_p0(cond)
        x = torch.cat([x, input_cond.view(x.shape[0],-1,self.dim_pose_embed)], dim=-1)
        for i in range(len(self.input_layers)):
            x = self.input_layers[i](x)
            x = self.offset_act(x)  
        raw_sigma = self.sigma_layer(x)
        return raw_sigma        

class ImplicitNet(nn.Module):
    def __init__(self, opt, cano_cond=None):
        super().__init__()

        dims = [opt.d_in + opt.hash_d] + list(
            opt.dims) + [opt.d_out + opt.feature_vector_size]
        self.num_layers = len(dims)
        self.skip_in = opt.skip_in
        self.embed_fn = None
        self.opt = opt
        self.cano_cond = cano_cond
        self.n_kps = opt.kps

        if opt.multires > 0:
            embed_fn, input_ch = get_embedder(opt.multires, input_dims=opt.d_in, mode=opt.embedder_mode)
            self.embed_fn = embed_fn
            dims[0] = input_ch + opt.hash_d
        self.cond = opt.cond   
        self.cam_cond = opt.cam_cond
        self.sem_cond = opt.sem_cond 
        if self.cond in ['kps', 'cano']:
            self.cond_layer = [0]
            self.cond_dim = self.n_kps*3
        elif self.cond == 'frame':
            self.cond_layer = [0]
            self.cond_dim = opt.dim_frame_encoding

        if self.sem_cond != 'none':
            self.cond_layer = [0]
            self.sem_cond_dim = 3

        self.dim_pose_embed = 0
        if self.dim_pose_embed > 0:
            self.lin_p0 = nn.Linear(self.cond_dim, self.dim_pose_embed)
            self.cond_dim = self.dim_pose_embed
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0] #- opt.hash_d
            else:
                out_dim = dims[l + 1]
            
            if self.cond != 'none' and self.cam_cond != 'none' and self.sem_cond != 'none' and l in self.cond_layer: 
                lin = nn.Linear(dims[l] + self.cond_dim + self.sem_cond_dim + opt.dim_frame_encoding, out_dim)
            elif self.cond != 'none' and self.cam_cond != 'none' and self.sem_cond == 'none' and l in self.cond_layer: ### bg SDF (conditioned on frame and camera)
                lin = nn.Linear(dims[l] + self.cond_dim + opt.dim_frame_encoding, out_dim)                
            elif self.cond != 'none' and self.cam_cond == 'none' and self.sem_cond != 'none' and l in self.cond_layer: ### fg SDF (conditioned on pose and semantic)
                lin = nn.Linear(dims[l] + self.cond_dim + self.sem_cond_dim, out_dim)  
            elif self.cond != 'none' and self.cam_cond == 'none' and self.sem_cond == 'none' and l in self.cond_layer: ### fg SDF (conditioned on pose and semantic)
                lin = nn.Linear(dims[l] + self.cond_dim, out_dim)                 
            elif self.cond == 'none' and self.cam_cond == 'none' and self.sem_cond != 'none' and l in self.cond_layer: ### weight net (conditioned on semantic)
                lin = nn.Linear(dims[l] + self.sem_cond_dim, out_dim)                            
            else:
                lin = nn.Linear(dims[l], out_dim)
              
            if opt.init == 'geometry':
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.bias)
                elif opt.multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif opt.multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
            if opt.init == 'zero':
                init_val = 1e-5
                if l == self.num_layers - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.uniform_(lin.weight, -init_val, init_val)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, cond): 

        if input.ndim == 2: input = input.unsqueeze(0)
        num_batch, num_point, num_dim = input.shape
        if num_batch * num_point == 0: return input
        input = input.reshape(num_batch * num_point, num_dim)

        if self.cond != 'none':
            ######### assuming for joint condition (sdfnet) last dimension (kp*3)   
            num_batch, num_cond = cond[self.cond].shape
            input_cond = cond[self.cond].unsqueeze(1).expand(num_batch, num_point, num_cond)
            input_cond = input_cond.reshape(num_batch * num_point, num_cond)            
            #else: ######## for weights net (last dimension 1 == (semantic label))
            #    input_cond = cond[self.cond].reshape(num_batch * num_point, 1)

            if self.dim_pose_embed:
                input_cond = self.lin_p0(input_cond)

        if self.cam_cond != 'none':
            num_batch, num_cond = cond[self.cam_cond].shape
            input_cam_cond = cond[self.cam_cond].unsqueeze(1).expand(num_batch, num_point, num_cond)
            input_cam_cond = input_cond.reshape(num_batch * num_point, num_cond)

            if self.dim_pose_embed:
                input_cam_cond = self.lin_p0(input_cam_cond)               

        if self.sem_cond != 'none':
            input_sem_cond = cond[self.sem_cond].reshape(num_batch * num_point, self.sem_cond_dim) 

        if self.embed_fn is not None:
            input = self.embed_fn(input)        

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if self.cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)
            if self.cam_cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_cam_cond], dim=-1)
            if self.sem_cond != 'none' and l in self.cond_layer:
                x = torch.cat([x, input_sem_cond], dim=-1)  
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        
        x = x.reshape(num_batch, num_point, -1)

        return x

    def gradient(self, x, cond):
        x.requires_grad_(True)
        y = self.forward(x, cond)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.mode = opt.mode
        self.num_sem = opt.num_sem
        self.ao_layer = opt.ao_layer
        self.net_depth_condition = 1  
        self.n_kps = opt.kps      
        dims = [opt.d_in + opt.feature_vector_size] + list(
            opt.dims) + [opt.d_out]

        self.embedview_fn = None
        if opt.multires_view > 0:
            embedview_fn, input_ch = get_embedder(opt.multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        if self.mode == 'nerf_frame_encoding':
            dims[0] += 2*opt.dim_frame_encoding ### included camera encoding also
        if self.mode == 'pose':
            self.dim_cond_embed = 8 
            self.cond_dim = self.n_kps*3 # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            dims[0] += opt.dim_frame_encoding
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        if self.ao_layer:
            self.dim_cond_embed = 8
            self.cond_dim = self.n_kps * 3
            self.net_width_condition = 128
            self.num_ao_channels = 1
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
            self.bottleneck_ao_layer = dense_layer(
                256, 256
            )
            self.condition_ao_layers = nn.ModuleList()
            in_features = 256 + self.dim_cond_embed
            for _ in range(self.net_depth_condition):
                self.condition_ao_layers.append(
                    dense_layer(in_features, self.net_width_condition)
                )
                in_features = self.net_width_condition            
            self.ao_layer_act = dense_layer(in_features, self.num_ao_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if self.num_sem > 0:
            self.lin_semhead = dense_layer(
                256, self.num_sem
            )
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, cam_latent_code=None, frame_latent_code=None):
        if self.embedview_fn is not None:
            if self.mode == 'nerf_frame_encoding':
                view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'nerf_frame_encoding':
            frame_latent_code = frame_latent_code.expand(view_dirs.shape[0], -1)
            cam_latent_code = cam_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([view_dirs, frame_latent_code, cam_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'pose':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)
            cam_latent_code = cam_latent_code.expand(view_dirs.shape[0], -1)
            rendering_input = torch.cat([points, normals, body_pose, cam_latent_code, feature_vectors], dim=-1)
        elif self.mode == 'none':
            num_points = points.shape[0]
            body_pose = body_pose.unsqueeze(1).expand(-1, num_points, -1).reshape(num_points, -1)
            body_pose = self.lin_pose(body_pose)            
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
                raw_feat_sem = x
                if self.ao_layer:
                    raw_feat = x #####like TAVA
           
        x = self.sigmoid(x)

        if self.num_sem>0:
            raw_feat_sem = self.lin_semhead(raw_feat_sem)
            sem = self.softmax(raw_feat_sem)
        else:
            sem = None

        ##### pass raw_feat through ao_layer (ambient occlusion)
        if self.ao_layer:
            y = self.bottleneck_ao_layer(raw_feat)
            y = torch.cat([y, body_pose], dim=-1)
            for i in range(self.net_depth_condition):
                y = self.condition_ao_layers[i](y)
                y = self.relu(y)        
            raw_ao = self.ao_layer_act(y)   
            x = x * self.sigmoid(raw_ao)
        return x, sem
