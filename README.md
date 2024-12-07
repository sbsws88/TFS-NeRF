# TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scene


> [**TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scene**](https://github.com/sbsws88/tfs_nerf.github.io/)            
> Monash University, IIT Bombay   
> Sandika Biswas, Qianyi Wu, Biplab Banerjee, and Hamid Rezatofighi   

### [**[Project Page]**](https://github.com/sbsws88/tfs_nerf.github.io/) **|** [**[Paper]**](https://arxiv.org/abs/2409.17459)

Abstract
-----------------
Despite advancements in Neural Implicit models for 3D surface reconstruction, handling dynamic environments with interactions between arbitrary rigid, non-rigid, or deformable entities remains challenging. The generic reconstruction methods adaptable to such dynamic scenes often require additional inputs like depth or optical flow or rely on pre-trained image features for reasonable outcomes. These methods typically use latent codes to capture frame-by-frame deformations. Another set of dynamic scene reconstruction methods, are entity-specific, mostly focusing on humans, and relies on template models. In contrast, some template-free methods bypass these requirements and adopt traditional LBS (Linear Blend Skinning) weights for a detailed representation of deformable object motions, although they involve complex optimizations leading to lengthy training times. To this end, as a remedy, this paper introduces TFS-NeRF, a template-free 3D semantic NeRF for dynamic scenes captured from sparse or single-view RGB videos, featuring interactions among two entities and more time-efficient than other LBS-based approaches. Our framework uses an Invertible Neural Network (INN) for LBS prediction, simplifying the training process. By disentangling the motions of interacting entities and optimizing per-entity skinning weights, our method efficiently generates accurate, semantically separable geometries. Extensive experiments demonstrate that our approach produces high-quality reconstructions of both deformable and non-deformable objects in complex interactions, with improved training efficiency compared to existing methods. 

Getting started
-----------------
We recommend using conda to create separate Python environment.
```bash
# Create new conda env from given environment file
conda env create -f tfs_env.yml
conda activate tfs

```

Checkpoints
-----------------
Please download the pre-trained models for BEHAVE dataset from Google Drive **link** (https://drive.google.com/drive/folders/1Gh_d3yl1V3aP39s1cchJ1tw1Wh3bFILD?usp=sharing).

Dataset
-----------------

Please download the pre-processed dataset from the given **link** (https://drive.google.com/drive/folders/1txJDwKpd8xNgXB-ZUjsIoqzprPr4U-L1?usp=sharing) 

Run
-----------------

```bash
python reconstruct.py
```

Folder Format
-----------------
```
tfs_nerf
├── code
├── data
│   └── Date01_Sub01_basketball
│   	├── image
│       │   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── mask
│      	│   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── semantic_map
│      	│   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── keyframe_info
│   	├── cameras_normalize
│   	├── poses
│   └── Date01_Sub01_boxlarge_hand
│   	├── image
│       │   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── mask
│      	│   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── semantic_map
│      	│   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│   	├── keyframe_info
│   	├── cameras_normalize
│   	├── poses
├── outputs
│   └── Date01_Sub01_basketball
│   	├── model_w_realnvp
│   	    ├── checkpoints
│             ├── last.ckpt
│   └── Date01_Sub01_boxlarge_hand
│   	├── model_w_realnvp
│   	    ├── checkpoints
│             ├── last.ckpt
```

Citation
---------------
If you find TFS-NeRF useful in your research please consider citing:
```
@article{biswas2024tfs,
  title={TFS-NeRF: Template-Free NeRF for Semantic 3D Reconstruction of Dynamic Scene},
  author={Biswas, Sandika and Wu, Qianyi and Banerjee, Biplab and Rezatofighi, Hamid},
  journal={NeurIPS},
  year={2024}
}
```

## Acknowledgments
- Thanks to [Vid2Avatar](https://github.com/MoyGcc/vid2avatar/tree/main), [TAVA](https://github.com/facebookresearch/tava), [ObjectSDF](https://github.com/QianyiWu/objsdf), for their public code and released models. Our implementation was mainly inspired by the above repositories.
