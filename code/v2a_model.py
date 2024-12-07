import pytorch_lightning as pl
from lib.model.v2a import V2A
import torch
import hydra
import os
from datetime import datetime
from lib.utils.meshing import generate_mesh
import trimesh
from lib.model.real_nvp import skinning
from torch.utils.tensorboard import SummaryWriter

class V2AModel(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_training_frames = opt.dataset.metainfo.n_frames
        num_cams = opt.dataset.metainfo.n_cams
        self.keyframe_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.metainfo.data_dir, 'keyframe_info.npy')
        self.gender = opt.dataset.metainfo.gender
        self.model = V2A(opt.model, self.keyframe_path, self.gender, num_training_frames, num_cams)
        self.start_frame = opt.dataset.metainfo.start_frame
        self.end_frame = opt.dataset.metainfo.end_frame
        self.training_modules = ["model"]

        self.training_indices = list(range(self.start_frame, self.end_frame))

        self.sem_class = opt.model.implicit_network.d_out
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.writer = SummaryWriter(os.path.join(self.timestamp)) 

        torch.manual_seed(412)
        self.colorbases = torch.rand((opt.model.implicit_network.kps, 3)) * 255

    def query_oc(self, x, cond, sem_ind):
        
        x = x.reshape(-1, 3)
        sem_label = self.model.get_sem_label(x, self.model.jnts_v_cano)
        cond['sem_label'] = sem_label
        if sem_ind == 0:
            cond['kps'] = cond['jnts']
            mnfld_pred = self.model.implicit_network(x.unsqueeze(0), cond)[:,:,0].reshape(-1,1)
        elif sem_ind == 1:
            cond['kps'] = cond['jnts_obj']
            mnfld_pred = self.model.implicit_network_obj(x.unsqueeze(0), cond)[:,:,1].reshape(-1,1)        
        return {'sdf':mnfld_pred}

    def get_deformed_mesh_fast_mode(self, verts, tfs, jnts, indx, cond=None):
        verts = torch.tensor(verts).cuda().float()
        sem_cond = self.model.get_sem_label(verts, jnts)

        if indx == 0:
            weights, _, _ = self.model.deformer.query_weights(verts, sem_cond, cond) 
            tfs = tfs[:,:24]  
        else:
            weights, _, _ = self.model.deformer_obj.query_weights(verts, sem_cond, cond)  
            tfs = tfs[:,24:]   

        offset_world = None
        verts_deformed = skinning(verts.unsqueeze(0),  weights, tfs, offsets_world=offset_world).data.cpu().numpy()[0]
        return verts_deformed, weights
    
    def test_step(self, batch, *args, **kwargs):

        inputs, _, _, _, idx = batch
        tfs = inputs["tfs"]    
        cano_jnts = self.model.jnts_v_cano 

        cond = {}
        cond['jnts'] = (cano_jnts[:,:24,:]).reshape(cano_jnts.shape[0],-1) 
        cond['jnts_obj'] = (cano_jnts[:,24:,:]).reshape(cano_jnts.shape[0],-1)

        os.makedirs("test_mask", exist_ok=True)
        os.makedirs("test_rendering", exist_ok=True)
        os.makedirs("test_fg_rendering", exist_ok=True)
        os.makedirs("test_normal", exist_ok=True)

        save_dir = 'test_mesh'
        os.makedirs(save_dir, exist_ok=True)
        self.sem_class = 2
        for indx in range(0,self.sem_class):
            if indx==2:
                continue

            level_set = 0.0 if indx == 0 else 0.0
            mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, indx), point_batch=10000, res_up=3, level_set=level_set)

            verts_deformed, weights = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, tfs, cano_jnts, indx, self.model.jnts_v_cano)
            colors = self.colorbases[weights.argmax(dim=-1)] 
            mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, vertex_colors=colors[0].cpu().numpy())      
            mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces, vertex_colors=colors[0].cpu().numpy())         
            
            mesh_canonical.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_{int(indx)}_canonical.ply")
            mesh_deformed.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_{int(indx)}_deformed.ply")
