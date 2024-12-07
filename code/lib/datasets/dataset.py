import os
import glob
import hydra
import cv2
import imageio
import numpy as np
import torch
from lib.utils import utils_sephead


class Dataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        root = os.path.join(metainfo.root, metainfo.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = metainfo.start_frame
        self.end_frame = metainfo.end_frame 
        self.skip_step = 1
        #self.images, self.img_sizes = [], []
        self.training_indices = list(range(metainfo.start_frame, metainfo.end_frame, self.skip_step))

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))

        #import pdb; pdb.set_trace()

        # only store the image paths to avoid OOM
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        self.n_images = len(self.img_paths)

        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]

        # semantic map, for semantic supervision
        semantic_dir = os.path.join(root, "semantic_map")
        self.semantic_paths = sorted(glob.glob(f"{semantic_dir}/*.png"))  
        self.semantic_paths = [self.semantic_paths[i] for i in self.training_indices]    

        keyframe_info = np.load(os.path.join(root, 'keyframe_info.npy'),allow_pickle=True).item()
        self.tfs = keyframe_info['tfs'][self.training_indices] #[:,24:,:,:]

        joints_info = np.load(os.path.join(root, 'poses.npy')) #, allow_pickle = True).item()
        self.jnts = joints_info[self.training_indices] #[:,24:,:] #['pose']

        self.meta_info = np.load(os.path.join(root, 'meta_info.npy'))

        # cameras
        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.training_indices]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.training_indices]

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = utils_sephead.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all)

        # other properties
        self.num_sample = split.num_sample
        self.sampling_strategy = "weighted"

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # normalize RGB
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255

        mask_img = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY) > 0
        #semantic_map = cv2.imread(self.semantic_paths[idx])[:,:,0] #mask_img/255.0 #
        semantic_map = imageio.imread(self.semantic_paths[idx])[:,:,0] #mask_img/255.0 #

        #semantic_map = semantic_map[:,:,0]

        img_size = self.img_size

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        tfs = torch.from_numpy(self.tfs[idx]).float()
        jnts = torch.from_numpy(self.jnts[idx]).float()

        if self.num_sample > 0:   ##### for training num_sample 512, for test and validation num_sample -1 (means all pixels)
            data = {
                "rgb": img,
                "uv": uv,
                "object_mask": mask,
                "semantic_map": semantic_map
            }
            #print(self.img_paths[idx])
            samples, index_outside, ind_inside_human, indices, n_sample_h = utils_sephead.obj_aware_weighted_sampling(data, img_size, self.num_sample)
            inputs = {
                "rgb":img,                
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "tfs": tfs,
                "jnts": jnts,
                'index_outside': index_outside,
                'index_inside_human': ind_inside_human,
                "idx": idx,
                "all_ind": indices,
                "meta_info": self.meta_info[idx],
                "semantic": samples["semantic_map"], #.astype(np.float32) #/3.0
                "n_sample_h":n_sample_h
            }
            images = {
                "rgb": samples["rgb"].astype(np.float32),
                "semantic": samples["semantic_map"]} #.astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "tfs": tfs,
                "jnts": jnts,
                "idx": idx
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32),
                "semantic" : semantic_map.reshape(-1, 1), #.astype(np.float32),
                "img_size": self.img_size
            }
            return inputs, images

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)
        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))  
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize

        img = img[:, :, ::-1] / 255        
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            'image_id': image_id,
            "tfs": inputs["tfs"],
            "jnts": inputs["jnts"],
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            "semantic": images["semantic"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': self.total_pixels
        }
        return inputs, images

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, metainfo, split):
        self.dataset = Dataset(metainfo, split)

        self.img_size = self.dataset.img_size

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = split.pixel_per_batch
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]       

        inputs, images = data
        inputs = {
            "uv": inputs["uv"],
            "rgb": images["rgb"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "tfs": inputs["tfs"],
            "jnts": inputs["jnts"],
            "idx": inputs['idx'],
            "current_epoch": 100,
        }
        images = {
            "rgb": images["rgb"],
            "img_size": images["img_size"]
        }
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx
