# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root of the kubric module."""

# --- auto-computed by setup.py, source version is always at HEAD
__version__ = "HEAD"

from kubric import assets
from kubric.assets import AssetSource
from kubric.core.assets import Asset, UndefinedAsset
from kubric.core.cameras import Camera, OrthographicCamera, PerspectiveCamera, UndefinedCamera
from kubric.core.color import Color, get_color
from kubric.core.lights import DirectionalLight, Light, PointLight, RectAreaLight, SpotLight, UndefinedLight
from kubric.core.materials import FlatMaterial, Material, PrincipledBSDFMaterial, Texture, UndefinedMaterial
from kubric.core.objects import Cube, FileBasedObject, Object3D, PhysicalObject, Sphere
from kubric.core.scene import Scene
from kubric.file_io import (
    as_path,
    read_png,
    read_tiff,
    write_image_dict,
    write_json,
    write_palette_png,
    write_pkl,
    write_png,
    write_scaled_png,
    write_tiff,
)
from kubric.kubric_typing import AddAssetFunction, PathLike
from kubric.post_processing import adjust_segmentation_idxs, compute_bboxes, compute_visibility
from kubric.randomness import (
    move_until_no_overlap,
    position_sampler,
    random_hue_color,
    random_rotation,
    resample_while,
    rotation_sampler,
    sample_point_in_half_sphere_shell,
)
from kubric.utils import (
    ArgumentParser,
    done,
    get_camera_info,
    get_instance_info,
    get_scene_metadata,
    log_my_flags,
    process_collisions,
    setup,
    setup_directories,
    setup_logging,
)

# --- basic kubric types
from pyquaternion import Quaternion
