import torch
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint
import logging
import open3d as o3d
import torch.nn.functional as tf
from pytorch3d import ops
from pytorch3d.transforms import quaternion_to_axis_angle
from lib.model.bone_utils import closest_distance_to_points

def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices
    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1] ** 2
    yy = quaternions[..., 2] ** 2
    zz = quaternions[..., 3] ** 2
    ww = quaternions[..., 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, mask):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.register_buffer("mask", mask)  # 1,1,1,3

    def forward(self, F, y):
        y1 = y * self.mask

        F_y1 = torch.cat([F, self.projection(y1)], dim=-1)
        s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)

        return x, ldj

    def inverse(self, F, x):
        x1 = x * self.mask
        F_x1 = torch.cat([F, self.projection(x1)], dim=-1)

        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        ldj = s.sum(-1)

        return y, ldj


class SimpleNVP(nn.Module):
    def __init__(
        self,
        n_layers,
        feature_dims,
        cano_skel,
        hidden_size,
        projection,
        skinning_net,
        offsetnet,
        checkpoint=True,
        normalize=True,
        explicit_affine=True,
        dist_type='bone',
        use_cond=None,
        n_kps=24,
        skel_type='non_rigid',
        softmax_type='softmax'
    ):
        super().__init__()
        self._checkpoint = checkpoint
        self._normalize = normalize
        self._explicit_affine = explicit_affine
        self._projection = projection
        self._create_layers(n_layers, feature_dims, hidden_size)
        self.skinning_net = skinning_net
        self.offsetnet = offsetnet
        self.soft_blend = 20
        self.dist_type = dist_type
        self.K=1
        self.n_init=1
        self.use_cond=use_cond
        self.max_dist=0.1
        self.n_kps=n_kps
        self.skel_type=skel_type
        self.softmax_type=softmax_type
        self.cano_skel = cano_skel

    def _create_layers(self, n_layers, feature_dims, hidden_size):
        input_dims = 3
        proj_dims = self._projection.proj_dims

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask = torch.zeros(input_dims)
            # mask[torch.arange(input_dims) % 2 == (i%2)] = 1
            #mask[torch.randperm(input_dims)[:2]] = 1
            if i ==0:
                mask[[0,2]] = 1
            else:
                mask[[0,1]] = 1

            logging.info("NVP {}th layer split is {}".format(i,mask))

            map_s = nn.Sequential(
                nn.Linear(proj_dims + feature_dims, hidden_size), #feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
                nn.Hardtanh(min_val=-10, max_val=10),
            )
            map_t = nn.Sequential(
                nn.Linear(proj_dims + feature_dims, hidden_size), #feature_dims, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dims),
            )
            self.layers.append(
                CouplingLayer(map_s, map_t, self._projection, mask[None, None, None])
            )

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3)
            )

        if self._explicit_affine:
            self.rotations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 4)
            )
            self.translations = nn.Sequential(
                nn.Linear(feature_dims, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 3)
            )
            self.skts = nn.Sequential(
                nn.Linear(9, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 16) 
            )            

    def _check_shapes(self, F, x):
        B1, M1, _ = F.shape  # batch, templates, C
        B2, _, M2, D = x.shape  # batch, Npts, templates, 3
        assert B1 == B2 and M1 == M2 and D == 3

    def _expand_features(self, F, x):
        _, N, n_init, _ = x.shape
        return F[:, None, None].expand(-1, N, n_init, -1)

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1
        sigma = sigma[:, None]

        return 0, sigma

    def _affine_input(self, F, y):
        if not self._explicit_affine:
            return torch.eye(3)[None, None, None].to(F.device), 0

        q = self.rotations(F) #.view(-1,(1),4)   
        q = q / torch.sqrt((q ** 2).sum(-1, keepdim=True))
        R = quaternions_to_rotation_matrices(q)[:, None]
        R_aa = quaternion_to_axis_angle(q)
        t = self.translations(F)[:, None] #.view(-1,(1),3) #[:, None]

        return R, R_aa, t

    def init_cano(self, y, F, tfs, inverse=True):
        """Transform x to canonical space for initialization

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
            joints_world (tensor): skeleton joints in the observation space

        Returns:
            xc_init (tensor): gradients. shape: [B, N, I, D]
        """

        n_batch, n_point, _ = y.shape
        _, n_joint, _, _ = tfs.shape

        if inverse == False: #### points are in canonical space
            F = self.cano_skel

        #### find nearest bones in world coord for initializing xc_init
        if self.dist_type == 'bone':
            dists = closest_distance_to_points(F, y, skel_type=self.skel_type)
            knn = torch.topk(dists, self.n_init, dim=-1, largest=False)
            knn_idxs = knn.indices + 1 #considering the root joint back   
        else:
            dists, knn_idxs, _ = ops.knn_points(y, F,
                                                K=self.K, return_nn=True) 

        xc_init = []        
        for i in range(knn_idxs.shape[-1]):
            w = torch.zeros((n_point, n_joint), device=y.device)
            ind = knn_idxs[:,:,i].squeeze(0).unsqueeze(-1)
            w[torch.arange(w.size(0)),ind.t()] = 1.
            xc_init.append(skinning(y, w.unsqueeze(0), tfs, inverse=inverse))

        xc_init = torch.stack(xc_init, dim=2)
        return xc_init, dists        

    def forward(self, F, x, sem_label, cam_cond, tfs, inverse=True):

        #### transforming canonical points to deformed points
        x_init, _ = self.init_cano(x, F, tfs, inverse)
        y = x_init
        F = F.reshape(F.shape[0],-1)
        #F = torch.cat([F, cam_cond], -1)
        F = self._expand_features(F, y) #.reshape(B,N,Nj,-1)
        F = torch.cat([F, sem_label.unsqueeze(-2)], -1)

        ldj = 0
        for l in self.layers:
            y, ldji = self._call(l, F, y)
            ldj = ldj + ldji
        return y, x_init    

    
    def get_weight(self, x, obj_type):

        B, N, _, ndim = x.shape 
        x = x.reshape(B, -1, ndim)
        cond={'kps':cond} if self.use_cond != 'none' else 'none'
        weights, _ = self.query_weights(x, obj_type, cond) 
        return weights #xd_opt, x.reshape(B, N, self.n_init, ndim)   
        

    def inverse(self, F, y, sem_label, tfs=None):

        cond = F.reshape(F.shape[0],-1)
        if self.use_cond == 'none':
            distance_batch, index_batch, neighbor_points = ops.knn_points(y, cond.reshape(1,self.n_kps,3),
                                                                        K=self.K, return_nn=True)
            distance_batch = torch.clamp(distance_batch, max=4)
            weights_conf = torch.exp(-distance_batch)
            distance_batch = torch.sqrt(distance_batch)
            weights_conf = weights_conf / weights_conf.sum(-2, keepdim=True) 
            outlier_mask = (distance_batch[..., 0] > self.max_dist)[0]   
        else:
            weights_conf = None
            outlier_mask = None

        y_init, _ = self.init_cano(y, F, tfs)

        x = y_init
        F = F.reshape(F.shape[0],-1)
        F = self._expand_features(F, x) #.reshape(B,N,Nj,-1)
        F = torch.cat([F, sem_label.unsqueeze(-2)], -1)       

        ldj = 0 
        for l in reversed(self.layers):
            x, ldji = self._call(l.inverse, F, x)
            ldj = ldj + ldji

        B, N, _, ndim = x.shape 
        x = x.reshape(B, -1, ndim)
        y = y[:,:,None].expand(-1, -1, self.n_init, -1)
        y = y.reshape(B, -1, ndim)
        cond={'kps':cond} if self.use_cond != 'none' else 'none'

        if self.offsetnet is not None:
            offset = self.offsetnet(x.reshape(B, -1, ndim), F)
        else:
            offset = None

        weights, sem_pred, _ = self.query_weights(x, sem_label, cond=cond) 
        if  weights_conf is not None:
              weights_conf = weights_conf.unsqueeze(-2).repeat(1,1,self.n_init,1).reshape(weights_conf.shape[0],-1,weights_conf.shape[-1])
              weights = weights * weights_conf.detach()
              weights = weights / weights.sum(-1, keepdim=True)

        xd_opt = skinning(y, weights, tfs, inverse=True, offsets_world=offset) 
        xd_opt = xd_opt.reshape(B, N, self.n_init, ndim)   

        return xd_opt[:,:,:,:3], x.reshape(B, N, self.n_init, ndim), outlier_mask, y_init, sem_pred #x.squeeze(-2),    
    
    def forward_skinning(self, xc, sem_cond, cond, tfs):
        weights, sem_pred, _ = self.query_weights(xc, sem_cond, cond=cond)
        x_transformed = skinning(xc, weights, tfs, inverse=False)
        return x_transformed    


    def query_weights(self, xc, sem_cond, cond=None):  
        weights_logit = self.skinning_net(xc, {'sem_label':sem_cond})
        weights = tf.softmax(weights_logit[:,:,:24] * self.soft_blend, dim=-1)
        sem_pred = tf.softmax(weights_logit[:,:,24:], dim=-1)        

        outlier_mask = None
        if cond != 'none' and self.use_cond == 'none':
            if len(xc.shape)==2:
                xc = xc.unsqueeze(0)
            distance_batch, index_batch, neighbor_points = ops.knn_points(xc, cond,
                                                                        K=self.K, return_nn=True)
            distance_batch = torch.clamp(distance_batch, max=4)
            weights_conf = torch.exp(-distance_batch)
            distance_batch = torch.sqrt(distance_batch)
            weights_conf = weights_conf / weights_conf.sum(-2, keepdim=True)
            weights = weights * weights_conf.detach() 
            weights = weights / weights.sum(-1, keepdim=True)
            outlier_mask = (distance_batch[..., 0] > self.max_dist)[0]  
        return weights, sem_pred, outlier_mask
   
    
    def gradient(self, xc, cond, tfs):
        """Get gradients df/dx

        Args:
            xc (tensor): canonical points. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            grad (tensor): gradients. shape: [B, N, D, D]
        """
        x = self.forward_skinning(xc, cond, tfs)

        grads = []
        for i in range(x.shape[-1]):
            d_out = torch.zeros_like(x, requires_grad=False, device=x.device)
            d_out[:, :, i] = 1
            grad = torch.autograd.grad(
                outputs=x,
                inputs=xc,
                grad_outputs=d_out,
                create_graph=False,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads.append(grad)

        return torch.stack(grads, dim=-2)       
    
    
def skinning(x, w, tfs, inverse=False, offsets_world=None, offsets_cano=None):
    """Linear blend skinning
    Args:
        x (tensor): canonical points. shape: [B, N, D]
        w (tensor): conditional input. [B, N, J]
        tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    """
    offsets_cano = 0.0 if offsets_cano is None else offsets_cano
    offsets_world = 0.0 if offsets_world is None else offsets_world    

    if inverse:
        x = x - offsets_world
        x_h = tf.pad(x, (0, 1), value=1.0)
        # p:n_point, n:n_bone, i,k: n_dim+1
        w_tf = torch.einsum("bpn,bnij->bpij", w, tfs)
        x_h = torch.einsum("bpij,bpj->bpi", w_tf.inverse(), x_h)[:, :, :3]
        x_h = x_h - offsets_cano
    else:
        x = x + offsets_world
        x_h = tf.pad(x, (0, 1), value=1.0)
        x_h = torch.einsum("bpn,bnij,bpj->bpi", w, tfs, x_h)[:, :, :3]
        x_h = x_h + offsets_cano
    return x_h    
