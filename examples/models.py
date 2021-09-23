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

PATH = os.path.dirname(os.path.realpath(__file__))
FACE_DETECTION_MODEL = os.path.join(
    PATH, 'models', 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
OBJECT_DETECTION_MODEL = os.path.join(
    PATH, 'models', 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
CLASSIFICATION_MODEL = os.path.join(
    PATH, 'models', 'tf2_mobilenet_v2_1.0_224_ptq_edgetpu.tflite')
CLASSIFICATION_IMPRINTING_MODEL = os.path.join(
    PATH, 'models', 'mobilenet_v1_1.0_224_l2norm_quant_edgetpu.tflite')
MOVENET_MODEL = os.path.join(
    PATH, 'models', 'movenet_single_pose_lightning_ptq_edgetpu.tflite')

CLASSIFICATION_LABELS = os.path.join(PATH, 'models', 'imagenet_labels.txt')
OBJECT_DETECTION_LABELS = os.path.join(PATH, 'models', 'coco_labels.txt')