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
"""MoveNet pose classification example.

You must first complete the Google Colab to train the pose classification model:
https://colab.sandbox.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tutorials/pose_classification.ipynb
"""

from aiy.coral import vision
from pycoral.utils.dataset import read_label_file

MOVENET_CLASSIFY_MODEL='models/pose_classifier.tflite'
MOVENET_CLASSIFY_LABELS='models/pose_labels.txt'

pose_detector = vision.PoseDetector(vision.MOVENET_MODEL)
pose_classifier = vision.PoseClassifier(MOVENET_CLASSIFY_MODEL)
labels = read_label_file(MOVENET_CLASSIFY_LABELS)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    # Detect the body points and draw the skeleton
    pose = pose_detector.get_pose(frame)
    vision.draw_pose(frame, pose)
    # Classify different yoga poses
    label_id = pose_classifier.get_pose_type(pose)
    vision.draw_label(frame, labels.get(label_id))
    