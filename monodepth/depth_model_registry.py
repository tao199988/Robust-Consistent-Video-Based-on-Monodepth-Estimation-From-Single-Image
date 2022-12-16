#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

#from monodepth.depth_model import DepthModel
#from monodepth.midas_v2_model import MidasV2Model
from Adel_lib.multi_depth_model_woauxi import RelDepthModel
from Adel_lib.multi_depth_model_woauxi import DepthModel

from typing import List


def get_depth_model_list() -> List[str]:
    return ["midas2"]


def get_depth_model(type: str,backbone: str) -> DepthModel:
    if type == "midas2":
        return RelDepthModel(backbone=backbone)
    else:
        raise ValueError(f"Unsupported model type '{type}'.")




def create_depth_model(type: str) -> DepthModel:
    model_class = get_depth_model(type)
    return model_class()
