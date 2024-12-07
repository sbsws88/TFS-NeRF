import torch
import numpy as np


def _dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)


def get_end_points(joints, skel_type='human'):
    """Get the bone heads and bone tails."""

    if skel_type=='human':
        head_ind = [0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16, 17, 18, 19, 20, 21]
        tail_ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    if skel_type=='obj':
        head_ind = [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22] 
        tail_ind = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]                 
    if skel_type=='scene':
        head_ind = [0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16, 17, 18, 19, 20, 21,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44,45,46]
        tail_ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,29,30,31,33,34,35,37,38,39,41,42,43,45,46,47]            
    elif skel_type=='hand':
        head_ind = [0,1,2,0,4,5,0,7,8,0,10,11,0,13,14,0,17,0,18,0,20,0,22]
        tail_ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  
    elif skel_type=='hare':
        head_ind = [0,1,2,3,7,5,3,10,8,3,1,11,12,13,14,15,1,17,18,19,20,21,0,23,23,25,26,27,23,29,30,31,24,33]
        tail_ind = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

    heads = joints[:,head_ind,:]
    tails = joints[:,tail_ind,:]
    return heads, tails

def closest_distance_to_points(joints, points, skel_type='human'):
    """Cartesian distance from points to bones (line segments).

    https://zalo.github.io/blog/closest-point-between-segments/

    :params bones: [n_bones,]
    :params points: If individual is true, the shape should be [..., n_bones, 3].
        else, the shape is [..., 3]
    :returns distances [..., n_bones]
    """
    points = points[..., None, :]  # [..., 1, 3]
    heads, tails = get_end_points(joints, skel_type)
    t = _dot(points - heads, tails - heads) / _dot(tails - heads, tails - heads)
    p = heads + (tails - heads) * torch.clamp(t, 0, 1)
    dists = torch.linalg.norm(p - points, dim=-1)

    return dists

def get_bone_ids(skel_type='human'):

    if skel_type=='human':
        bone_ids = [[-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], 
                    [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], 
                    [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
    elif skel_type=='scene':
        bone_ids = [[-1, 0], [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], 
                    [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], 
                    [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23], 
                    [-1, 24], [24, 25], [25,26], [26,27], [-1,28], [28,29], [29,30],[30,31], 
                    [-1,32], [32,33], [33,34], [34,35], [-1,36], [36,37], [37,38], [38,39], 
                    [-1,40], [40,41], [41,42], [42,43], [-1,44], [44,45], [45,46], [46,47]]
    elif skel_type=='hand':
        bone_ids = [[-1, 0], [0, 16], [16, 17], [17, 1], [0, 18], [18, 19], [19, 4], [0, 20], [20, 21], [21, 10], [0, 22], [22, 23], [23, 7], [0, 13], [1, 2], [2, 3], [4, 5], [5, 6], [10, 11], [11, 12], [7, 8], [8, 9], [13, 14], [14, 15]]
    elif skel_type=='hare':
        bone_ids = [[-1,0], [0, 1], [0, 23], [1, 2], [2, 3], [3, 4], [3, 7], [3, 10], [7, 5], [5, 6], [10, 8], [8, 9], [23, 24], [24, 33], [33, 34], [23, 25], [25,26],[26,27],[27,28], [23,29], [29,30], [30,31], [31,32], [1,11], [11,12],[12,13],[13,14],[14,15],[15,16],[1,17], [17,18],[18,19],[19,20],[20,21],[21,22]]
    elif skel_type=='obj':
        bone_ids = [[-1,0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]]        
    return bone_ids