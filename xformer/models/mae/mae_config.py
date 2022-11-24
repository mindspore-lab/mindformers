from dataclasses import dataclass

from xformer.tools.register import XFormerRegister, XFormerModuleType


@XFormerRegister.register(XFormerModuleType.CONFIG)
@dataclass
class MaeConfig:
    mask_ratio: float = 0.75
    num_classes: int = 0
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    use_abs_pos_emb: bool = True
    decoder_layers: int = 8
    decoder_num_heads: int = 16
    decoder_dim: int = 512
    norm_pixel_loss: bool = True
    parallel_config: dict = None
    moe_config: dict = None
