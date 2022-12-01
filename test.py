from fixupunet.network import FixUpUnet
from omegaconf import DictConfig
import torch


try:
    print("#TEST - Instanciating 2D Unet")
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 5,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": True,
            "dim": 2,
        }
    )

    unet = FixUpUnet(cfg).to("cuda")
    print("#SUCCESS - Instanciating 2D Unet")
    x = torch.rand([1, 3, 256, 256], device="cuda")
    print("#TEST - Running 2D Unet")
    y = unet(x)
    print(
        f"#SUCCESS - Running 2D Unet with input size {x.shape} and output size {y.shape}"
    )

    print("#TEST - Instanciating 3D Unet")
    cfg = DictConfig(
        {
            "feat": 32,
            "in_feat": 3,
            "out_feat": 3,
            "down_layers": 4,
            "identity_layers": 3,
            "bottleneck_layers": 6,
            "skips": True,
            "act_fn": "relu",
            "out_act_fn": "none",
            "max_feat": 256,
            "script_submodules": False,
            "dim": 3,
        }
    )

    unet = FixUpUnet(cfg).to("cuda")
    print("#SUCCESS - Instanciating 3D Unet")
    x = torch.rand([1, 3, 32, 256, 256], device="cuda")
    print("#TEST - Running 3D Unet")
    y = unet(x)
    print(
        f"#SUCCESS - Running 3D Unet with input size {x.shape} and output size {y.shape}"
    )


except Exception as e:
    print("ERROR")
    print(e)
    quit()
