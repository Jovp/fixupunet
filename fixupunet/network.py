import sys

sys.path.insert(0, "./")
from torch import nn
import torch
from .modules import FixupConvModule, FixupResidualChain


class UnetFixUp(nn.Module):
    """
    Unet using residual blocks and residual chains without any normalization layer.
    Example of cfg to instanciate the network:

    from omegaconf import DictConfig
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
        }
    )


    """

    def __init__(self, cfg):
        super(UnetFixUp, self).__init__()

        feat = cfg.feat
        self.skip = cfg.skips
        max_feat = cfg.max_feat

        i = -1

        self.in_conv = torch.jit.script(
            FixupConvModule(cfg.in_feat, cfg.feat, 3, 1, True, "none", cfg.act_fn)
        )

        self.down_layers = nn.ModuleList()
        for i in range(cfg.down_layers):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i + 1) * feat, max_feat)
            self.down_layers.append(
                torch.jit.script(
                    FixupResidualChain(
                        feat_curr,
                        cfg.identity_layers,
                        3,
                        cfg.act_fn,
                        attention=0,
                        depth_init=2 * cfg.identity_layers,
                    )
                )
            )
            self.down_layers.append(
                torch.jit.script(
                    FixupConvModule(
                        feat_curr, feat_next, 4, 2, True, "none", "none", use_bias=True
                    )
                )
            )

        self.bottleneck_layers = nn.ModuleList()
        feat_curr = min(2 ** (i + 1) * feat, max_feat)
        self.bottleneck_layers.append(
            torch.jit.script(
                FixupResidualChain(
                    feat_curr, cfg.bottleneck_layers, 3, cfg.act_fn, attention=0
                )
            )
        )

        self.up_layers = nn.ModuleList()
        for i in range(cfg.down_layers, 0, -1):
            feat_curr = min(2**i * feat, max_feat)
            feat_next = min(2 ** (i - 1) * feat, max_feat)
            self.up_layers.append(nn.Upsample(scale_factor=2, mode="bilinear"))
            self.up_layers.append(
                torch.jit.script(
                    FixupConvModule(
                        feat_next + feat_curr,
                        feat_next,
                        1,
                        1,
                        False,
                        "none",
                        "none",
                        use_bias=True,
                    )
                )
            )
            self.up_layers.append(
                torch.jit.script(
                    FixupResidualChain(
                        feat_next,
                        cfg.identity_layers,
                        3,
                        cfg.act_fn,
                        attention=0,
                        depth_init=2 * cfg.identity_layers,
                    )
                )
            )

        self.out_conv = torch.jit.script(
            FixupConvModule(
                feat, cfg.out_feat, 3, 1, True, "none", cfg.out_act_fn, use_bias=True
            )
        )

    def forward(self, x):

        skips = []
        x = self.in_conv(x)

        for i, layer in enumerate(self.down_layers):
            x = layer(x)

            if isinstance(layer, FixupResidualChain) or (
                isinstance(layer, torch.jit.RecursiveScriptModule)
                and layer.original_name == "FixupResidualChain"
            ):
                skips.append(x)

        for layer in self.bottleneck_layers:
            x = layer(x)

        for i, layer in enumerate(self.up_layers):
            x = layer(x)

            if isinstance(layer, torch.nn.Upsample) or (
                isinstance(layer, torch.jit.RecursiveScriptModule)
                and layer.original_name == "Upsample"
            ):
                if self.skip:
                    x = torch.cat([x, skips.pop()], dim=1)

        return self.out_conv(x)
