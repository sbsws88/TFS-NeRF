import torch
import math
import cv2
import numpy as np

def masked_softmax(vec, mask, dim=-1, mode='softmax', soft_blend=1):
    if mode == 'softmax':

        vec = torch.distributions.Bernoulli(logits=vec).probs

        masked_exps = torch.exp(soft_blend*vec) * mask.float()
        masked_exps_sum = masked_exps.sum(dim)

        output = torch.zeros_like(vec)
        output[masked_exps_sum>0,:] = masked_exps[masked_exps_sum>0,:]/ masked_exps_sum[masked_exps_sum>0].unsqueeze(-1)

        output = (output * vec).sum(dim, keepdim=True)

        output = torch.distributions.Bernoulli(probs=output).logits

    elif mode == 'max':
        vec[~mask] = -math.inf
        output = torch.max(vec, dim, keepdim=True)[0]

    return output


''' Hierarchical softmax following the kinematic tree of the human body. Imporves convergence speed'''
def hierarchical_softmax(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 24, device=x.device)

    prob_all[:, [1, 2, 3]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [1, 2, 3]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [4, 5, 6]] = prob_all[:, [1, 2, 3]] * (sigmoid(x[:, [4, 5, 6]]))
    prob_all[:, [1, 2, 3]] = prob_all[:, [1, 2, 3]] * (1 - sigmoid(x[:, [4, 5, 6]]))

    prob_all[:, [7, 8, 9]] = prob_all[:, [4, 5, 6]] * (sigmoid(x[:, [7, 8, 9]]))
    prob_all[:, [4, 5, 6]] = prob_all[:, [4, 5, 6]] * (1 - sigmoid(x[:, [7, 8, 9]]))

    prob_all[:, [10, 11]] = prob_all[:, [7, 8]] * (sigmoid(x[:, [10, 11]]))
    prob_all[:, [7, 8]] = prob_all[:, [7, 8]] * (1 - sigmoid(x[:, [10, 11]]))

    prob_all[:, [12, 13, 14]] = prob_all[:, [9]] * sigmoid(x[:, [24]]) * softmax(x[:, [12, 13, 14]])
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [24]]))

    prob_all[:, [15]] = prob_all[:, [12]] * (sigmoid(x[:, [15]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [15]]))

    prob_all[:, [16, 17]] = prob_all[:, [13, 14]] * (sigmoid(x[:, [16, 17]]))
    prob_all[:, [13, 14]] = prob_all[:, [13, 14]] * (1 - sigmoid(x[:, [16, 17]]))

    prob_all[:, [18, 19]] = prob_all[:, [16, 17]] * (sigmoid(x[:, [18, 19]]))
    prob_all[:, [16, 17]] = prob_all[:, [16, 17]] * (1 - sigmoid(x[:, [18, 19]]))

    prob_all[:, [20, 21]] = prob_all[:, [18, 19]] * (sigmoid(x[:, [20, 21]]))
    prob_all[:, [18, 19]] = prob_all[:, [18, 19]] * (1 - sigmoid(x[:, [20, 21]]))

    prob_all[:, [22, 23]] = prob_all[:, [20, 21]] * (sigmoid(x[:, [22, 23]]))
    prob_all[:, [20, 21]] = prob_all[:, [20, 21]] * (1 - sigmoid(x[:, [22, 23]]))

    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

def hierarchical_softmax_hand(x):
    def softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def sigmoid(x):
        return torch.sigmoid(x)

    n_batch, n_point, n_dim = x.shape
    x = x.flatten(0,1)

    prob_all = torch.ones(n_batch * n_point, 21, device=x.device)

    prob_all[:, [7, 10, 4, 1, 13]] = prob_all[:, [0]] * sigmoid(x[:, [0]]) * softmax(x[:, [7, 10, 4, 1, 13]])
    prob_all[:, [0]] = prob_all[:, [0]] * (1 - sigmoid(x[:, [0]]))

    prob_all[:, [2]] = prob_all[:, [1]] * (sigmoid(x[:, [2]]))
    prob_all[:, [1]] = prob_all[:, [1]] * (1 - sigmoid(x[:, [2]]))

    prob_all[:, [3]] = prob_all[:, [2]] * (sigmoid(x[:, [3]]))
    prob_all[:, [2]] = prob_all[:, [2]] * (1 - sigmoid(x[:, [3]]))

    prob_all[:, [17]] = prob_all[:, [3]] * (sigmoid(x[:, [17]]))
    prob_all[:, [3]] = prob_all[:, [3]] * (1 - sigmoid(x[:, [17]]))

    prob_all[:, [14]] = prob_all[:, [13]] * (sigmoid(x[:, [14]]))
    prob_all[:, [13]] = prob_all[:, [13]] * (1 - sigmoid(x[:, [14]]))

    prob_all[:, [15]] = prob_all[:, [14]] * (sigmoid(x[:, [15]]))
    prob_all[:, [14]] = prob_all[:, [14]] * (1 - sigmoid(x[:, [15]]))    

    prob_all[:, [16]] = prob_all[:, [15]] * (sigmoid(x[:, [16]]))
    prob_all[:, [15]] = prob_all[:, [15]] * (1 - sigmoid(x[:, [16]]))

    prob_all[:, [5]] = prob_all[:, [4]] * (sigmoid(x[:, [5]]))
    prob_all[:, [4]] = prob_all[:, [4]] * (1 - sigmoid(x[:, [5]]))

    prob_all[:, [6]] = prob_all[:, [5]] * (sigmoid(x[:, [6]]))
    prob_all[:, [5]] = prob_all[:, [5]] * (1 - sigmoid(x[:, [6]]))    

    prob_all[:, [18]] = prob_all[:, [6]] * (sigmoid(x[:, [18]]))
    prob_all[:, [6]] = prob_all[:, [6]] * (1 - sigmoid(x[:, [18]]))    

    prob_all[:, [11]] = prob_all[:, [10]] * (sigmoid(x[:, [11]]))
    prob_all[:, [10]] = prob_all[:, [10]] * (1 - sigmoid(x[:, [11]]))

    prob_all[:, [12]] = prob_all[:, [11]] * (sigmoid(x[:, [12]]))
    prob_all[:, [11]] = prob_all[:, [11]] * (1 - sigmoid(x[:, [12]]))    

    prob_all[:, [19]] = prob_all[:, [12]] * (sigmoid(x[:, [19]]))
    prob_all[:, [12]] = prob_all[:, [12]] * (1 - sigmoid(x[:, [19]]))        


    prob_all[:, [8]] = prob_all[:, [7]] * (sigmoid(x[:, [8]]))
    prob_all[:, [7]] = prob_all[:, [7]] * (1 - sigmoid(x[:, [8]]))

    prob_all[:, [9]] = prob_all[:, [8]] * (sigmoid(x[:, [9]]))
    prob_all[:, [8]] = prob_all[:, [8]] * (1 - sigmoid(x[:, [9]]))    

    prob_all[:, [20]] = prob_all[:, [9]] * (sigmoid(x[:, [20]]))
    prob_all[:, [9]] = prob_all[:, [9]] * (1 - sigmoid(x[:, [20]]))  


    prob_all = prob_all.reshape(n_batch, n_point, prob_all.shape[-1])
    return prob_all

def rectify_pose(pose, root_abs):
    """
    Rectify AMASS pose in global coord adapted from https://github.com/akanazawa/hmr/issues/50.
 
    Args:
        pose (72,): Pose.

    Returns:
        Rotated pose.
    """
    pose = pose.copy()
    R_abs = cv2.Rodrigues(root_abs)[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = np.linalg.inv(R_abs).dot(R_root)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose