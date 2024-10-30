# TFS-NeRF

# Abstract
Despite advancements in Neural Implicit models for 3D surface reconstruction,
handling dynamic environments with interactions between arbitrary rigid, non-
rigid, or deformable entities remains challenging. Many template-based methods
are entity-specific, focusing on humans, while . The generic reconstruction meth-
ods adaptable to such dynamic scenes often require additional inputs like depth
or optical flow or rely on pre-trained image features for reasonable outcomes.
These methods typically use latent codes to capture frame-by-frame deformations.
Another set of dynamic scene reconstruction methods, are entity-specific, mostly
focusing on humans, and relies on template models. In contrast, some template-free
methods bypass these requirements and adopt traditional LBS (Linear Blend Skin-
ning) weights for a detailed representation of deformable object motions, although
they involve complex optimizations leading to lengthy training times. To this end,
as a remedy, this paper introduces TFS-NeRF, a template-free 3D semantic NeRF
for dynamic scenes captured from sparse or single-view RGB videos, featuring
interactions among various two entities and more time-efficient than other LBS-
based approaches. Our framework uses an Invertible Neural Network (INN) for
LBS prediction, simplifying the training process. By disentangling the motions of
multiple interacting entities and optimizing per-entity skinning weights, our method
efficiently generates accurate, semantically separable geometries. Extensive experi-
ments demonstrate that our approach produces high-quality reconstructions of both
deformable and non-deformable objects in complex interactions, with improved
training efficiency compared to existing methods.
