import os

from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SiglipVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower)
    if  vision_tower.startswith("openai") or vision_tower.startswith("laion") or 'clip' in vision_tower:
        vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif 'sig' in vision_tower:
        vision_tower = SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
    return vision_tower 
