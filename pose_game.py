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
"""MoveNet pose estimation example."""

import argparse
from aiy.coral import vision
from pycoral.adapters.detect import BBox

RIGHT_WRIST = int(vision.KeypointType.RIGHT_WRIST)
LEFT_WRIST = int(vision.KeypointType.LEFT_WRIST)

# BGR (not RGB)
RED = (0, 0, 255)
GREEN = (0, 255, 0) 

def is_point_in_box(point, bbox):
    """
    Check if the given (x,y) point lies within the given box.

    Args:
      point (int tuple): The (x,y)-coordinates for the point
      bbox (BBox): A `BBox` (bounding box) object
    Returns:
      True if the point is inside the bounding box; False otherwise
    """
    x,y = point
    if (x > bbox.xmin and x < bbox.xmax) and (y > bbox.ymin and y < bbox.ymax):
        return True
    return False


width, height = vision.VIDEO_SIZE
xmin = 0
ymin = 0
xmax = int(width * 0.5)
ymax = int(height * 0.5)
fence = BBox(xmin, ymin, xmax, ymax)

pose_detector = vision.PoseDetector(vision.MOVENET_MODEL)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    pose = pose_detector.get_pose(frame)
    keypoints = vision.draw_pose(frame, pose)

    vision.draw_rect(frame, fence, color=RED)
    if RIGHT_WRIST in keypoints:
        if is_point_in_box(keypoints[RIGHT_WRIST], fence):
            vision.draw_rect(frame, fence, color=GREEN)
    