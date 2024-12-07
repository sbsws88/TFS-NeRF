import numpy as np
import torch
from skimage import measure
from lib.libmise import mise
import trimesh
import open3d as o3d

def generate_mesh(func, level_set=0.0, res_init=32, res_up=3, point_batch=5000):
    scale = 1.1  # Scale of the padded bbox regarding the tight one.
    
    gt_bbox = np.asarray([[-0.3643283,  -0.8319907,  -0.36400956],[ 0.28009942,  0.78449166,  0.36400977]]) 
    gt_center = (gt_bbox[0] + gt_bbox[1]) * 0.5
    gt_scale = (gt_bbox[1] - gt_bbox[0]).max()

    mesh_extractor = mise.MISE(res_init, res_up, level_set)
    points = mesh_extractor.query()

    # query occupancy grid
    while points.shape[0] != 0:
        
        orig_points = points
        points = points.astype(np.float32)
        points = (points / mesh_extractor.resolution - 0.5) * scale
        points = points * gt_scale + gt_center
        points = torch.tensor(points).float().cuda() 

        values = []
        for _, pnts in enumerate((torch.split(points,point_batch,dim=0))):
            out = func(pnts)
            values.append(out['sdf'].data.cpu().numpy())
        values = np.concatenate(values, axis=0).astype(np.float64)[:,0]        
        mesh_extractor.update(orig_points, values)        
        points = mesh_extractor.query()
    
    value_grid = mesh_extractor.to_dense()
    print(np.min(value_grid), np.max(value_grid))

    # marching cube
    verts, faces, normals, values = measure.marching_cubes_lewiner(
                                                volume=value_grid,
                                                gradient_direction='ascent',
                                                level=level_set)

    verts = (verts / mesh_extractor.resolution - 0.5) * scale
    verts = verts * gt_scale + gt_center
    faces = faces[:, [0,2,1]]

    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    #remove disconnect part
    connected_comp = meshexport.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport = max_comp

    return meshexport


