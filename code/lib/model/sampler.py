import torch

class PointOnBones:
    def __init__(self, bone_ids, global_sigma=0.02, local_sigma=0.005, skel_type='human'):
        self.bone_ids = bone_ids
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma        

        if skel_type == 'human':
            self.bone_for_global_sample = [0, 3, 6, 9]
        elif skel_type == 'obj':
            self.bone_for_global_sample = [0]   
        elif skel_type == 'scene':
            self.bone_for_global_sample = [0, 3, 6, 9, 24, 28, 32, 36]                      
        elif skel_type == 'hand':
            self.bone_for_global_sample = [0,16,17,18,19,20,21,22,23]
        elif skel_type == 'hare':
            self.bone_for_global_sample = [0, 1, 11, 17, 23, 24, 25, 12, 18, 25, 29] #[0, 3, 6, 9]

    def get_points(self, joints, joints_gt, num_per_bone=5):
        """Sample points on bones in canonical space.

        Args:
            joints (tensor): joint positions to define the bone positions. shape: [B, J, D]
            num_per_bone (int, optional): number of sample points on each bone. Defaults to 5.

        Returns:
            samples (tensor): sampled points in canoncial space. shape: [B, ?, 3]
            probs (tensor): ground truth occupancy for samples (all 1). shape: [B, ?]
        """

        num_batch, _, _ = joints.shape
        samples, samples_gt, semantic = [], [], []
        cnt = 0

        for bone_id in self.bone_ids:

            if bone_id[0] < 0 or bone_id[1] < 0:
                continue

            num_per_bone1 = num_per_bone*2 if bone_id[0] in  self.bone_for_global_sample else num_per_bone

            bone_dir = joints[:, bone_id[1]] - joints[:, bone_id[0]]
            bone_dir_gt = joints_gt[:, bone_id[1]] - joints_gt[:, bone_id[0]]

            scalars = (
                torch.linspace(0, 1, steps=num_per_bone1, device=joints.device)
                .unsqueeze(0)
                .expand(num_batch, -1)
            )
            scalars = (
                scalars + torch.randn((num_batch, num_per_bone1), device=joints.device) * 0.1
            ).clamp_(0, 1)
            
            # for center body sample points from larger dist
            sigma = self.global_sigma if bone_id[0] in  self.bone_for_global_sample else self.local_sigma

            # b: num_batch, n: num_per_bone, i: 3-dim            
            sample_points = joints[:, bone_id[0]].unsqueeze(1).expand(-1, scalars.shape[-1], -1) + torch.einsum("bn,bi->bni", scalars, bone_dir) 
            move_around_bone = torch.randn_like(sample_points) * sigma
            sample_points = sample_points + move_around_bone
            samples.append(sample_points)

            # b: num_batch, n: num_per_bone, i: 3-dim            
            sample_points_gt = joints_gt[:, bone_id[0]].unsqueeze(1).expand(-1, scalars.shape[-1], -1) + torch.einsum("bn,bi->bni", scalars, bone_dir_gt)
            sample_points_gt = sample_points_gt + move_around_bone            
            samples_gt.append(sample_points_gt) 

            sem_l = 1 if bone_id[0] < 24 else 2      
            cnt = cnt + 1

            semantic.append(sem_l*torch.ones(1, sample_points_gt.shape[1], 1))           

        samples = torch.cat(samples, dim=1)
        samples_gt = torch.cat(samples_gt, dim=1)  
        semantic = torch.cat(semantic, dim=1) 

        probs = torch.ones((num_batch, samples.shape[1]), device=joints.device)

        return samples, samples_gt, semantic, probs

    def get_joints(self, joints):
        """Sample joints in canonical space.

        Args:
            joints (tensor): joint positions to define the bone positions. shape: [B, J, D]

        Returns:
            samples (tensor): sampled points in canoncial space. shape: [B, ?, 3]
            weights (tensor): ground truth skinning weights for samples (all 1). shape: [B, ?, J]
        """
        num_batch, num_joints, _ = joints.shape

        samples = []
        weights = []

        for k in range(num_joints):
            samples.append(joints[:, k])
            weight = torch.zeros((num_batch, num_joints), device=joints.device)
            weight[:, k] = 1
            weights.append(weight)

        samples = torch.stack(samples, dim=1)
        weights = torch.stack(weights, dim=1)

        return samples, weights
    
    def get_points_on_bone(self, joints, range=(0.0, 1.0), num_per_bone=5):
        """Sample joints in canonical space.

        Args:
            joints (tensor): joint positions to define the bone positions. shape: [B, J, D]

        Returns:
            samples (tensor): sampled points in canoncial space. shape: [B, ?, 3]
            weights (tensor): ground truth skinning weights for samples (all 1). shape: [B, ?, J]
        """

        sample_joints, joints_weights = self.get_joints(joints)
        num_batch, num_joints, _ = joints.shape

        samples = []
        weights = []

        for bone_id in self.bone_ids:
            if bone_id[0] < 0 or bone_id[1] < 0:
                continue

            bone_dir = joints[:, bone_id[1]] - joints[:, bone_id[0]]

            t_vals = (
            torch.rand((num_batch, num_per_bone), device=joints.device)
            * (range[1] - range[0])
            + range[0])

            sample_points = (
            joints[:, bone_id[0]][None, :, :] + (joints[:, bone_id[1]] - joints[:, bone_id[0]])[None, :, :] * t_vals[:, :, None]
            )         

            samples.append(sample_points)

            weight = torch.zeros((num_batch, num_per_bone, num_joints), device=joints.device)
            weight[:, :, bone_id[0]] = 1
            weights.append(weight)

        samples = torch.cat(samples, dim=1)
        weights = torch.cat(weights, dim=1)

        return samples, weights    


class PointInSpace:
    def __init__(self, global_sigma=0.5, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma

    def get_points(self, pc_input=None, local_sigma=None, global_ratio=0.125):
        """Sample one point near each of the given point + 1/8 uniformly. 
        Args:
            pc_input (tensor): sampling centers. shape: [B, N, D]
        Returns:
            samples (tensor): sampled points. shape: [B, N + N / 8, D]
        """

        batch_size, sample_size, dim = pc_input.shape
        if local_sigma is None:
            sample_local = pc_input + (torch.randn_like(pc_input) * self.local_sigma)
        else:
            sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
        sample_global = (
            torch.rand(batch_size, int(sample_size * global_ratio), dim, device=pc_input.device)
            * (self.global_sigma * 2)
        ) - self.global_sigma

        sample = torch.cat([sample_local, sample_global], dim=1)

        return sample


