# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
from pycoral.adapters.detect import BBox
from coralkit import vision
import models

PATH = os.path.dirname(os.path.realpath(__file__))

POSENET_MODEL = os.path.join(PATH, 'models', 'posenet_mobilenet_v1_075_353_481_16_quant_decoder_edgetpu.tflite')
pose_detector = vision.PoseDetector(POSENET_MODEL, posenet=True)

for frame in vision.get_frames():
    poses = pose_detector.get_poses(frame)
    for pose in poses:
        vision.draw_pose(frame, pose[1])

