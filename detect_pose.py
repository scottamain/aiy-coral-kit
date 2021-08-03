# Lint as: python3
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
r"""Example using PyCoral to estimate a single human pose with Edge TPU MoveNet.

For more details about MoveNet and its best practices, please see
https://www.tensorflow.org/hub/tutorials/movenet
Example usage:
```
bash examples/install_requirements.sh

python3 examples/movenet_pose_estimation.py \
  --model test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite  \
  --input test_data/squat.bmp
```
"""

import argparse

import cv2 # move to vision.py
from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
from aiy.coral import vision

_NUM_KEYPOINTS = 17

MODEL='models/movenet_single_pose_lightning_ptq_edgetpu.tflite'

def get_body_points(img, interpreter):
  #resized_img = img.resize(common.input_size(interpreter), Image.ANTIALIAS)
  resized_img = cv2.resize(frame, common.input_size(interpreter), fx=0, fy=0,
                                                                interpolation=cv2.INTER_CUBIC)
  common.set_input(interpreter, resized_img)

  interpreter.invoke()

  pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
  print(pose)
  return pose


interpreter = make_interpreter(MODEL)
interpreter.allocate_tensors()

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
  points = get_body_points(frame, interpreter)
  
  #draw = ImageDraw.Draw(img)
  height, width, _ = frame.shape
  for i in range(0, _NUM_KEYPOINTS):
      score = points[i][2]
      if score > 0.3:
          y = int(points[i][0] * height)
          x = int(points[i][1] * width)
          cv2.circle(frame, (x,y), radius=5, color=(0, 0, 255), thickness=-1)


