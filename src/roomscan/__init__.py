"""roomscan — Monocular depth -> TSDF voxel grid -> 3D room reconstruction.

Layering (imports only flow downward):

    cli  ->  pipeline  ->  depth_sources / geometry / evaluation / export
                             |                  |
                           models             types
                             |
                           dataio  ->  types

`types` has no internal dependencies. Nothing below `pipeline` may import
`pipeline` or `cli`.
"""

__version__ = "0.1.0"
