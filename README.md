# UnetFixUp
Unet Implementation using residual block and fixup initialization
 ----
 
Unet using residual blocks and residual chains without any normalization layer.
Example of cfg to instanciate the network:
```
    from omegaconf import DictConfig
    cfg = DictConfig(
        {
            "feat": 32, #Number of features at the highest resolution
            "in_feat": 3, #Number of input channels
            "out_feat": 3, #Number of output channels
            "down_layers": 5, #Number of downsamplings
            "identity_layers": 3, #Number of residual blocks before and after bottleneck. Meaning for a value of 3, we have 6 residual blocks at each level with two convolutions each
            "bottleneck_layers": 6, #Number of residuals blocks for bottleneck
            "skips": True, #Skip connections
            "act_fn": "relu",
            "out_act_fn": "none", #Activation after the final layer, usually none
            "max_feat": 256, #We doubles features when downsampling but cap it to this value
            "script_submodules": True, #Scripting for faster more efficient network
        }
    )
```
