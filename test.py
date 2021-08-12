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

import os
import subprocess
import sys
import time
from aiy.coral import vision


def usb_accelerator_connected():
  if subprocess.run(["lsusb", "-d", "18d1:9302"], capture_output=True).returncode == 0:
    return True
  if subprocess.run(["lsusb", "-d", "1a6e:089a"], capture_output=True).returncode == 0:
    return True
  return False


if __name__ == '__main__':
    
  print('--- Checking required files ---')
  if not os.path.isfile(vision.CLASSIFICATION_MODEL):
    print('Downloading files...')
    subprocess.call(["bash", "install_requirements.sh"])
  print("Files okay.\n")

  print('--- Testing camera ---')
  TIME_LIMIT = 4
  start = time.time()
  for frame in vision.get_frames():
    elapsed = int(time.time() - start)
    print('Closing video in...', TIME_LIMIT - elapsed, end='\r')
    if (elapsed >= TIME_LIMIT):
      print('\nCamera okay.\n')
      break
    pass

  print('--- Testing USB Accelerator ---')

  if not usb_accelerator_connected():
    print('Coral USB Accelerator NOT found!')
    print('Make sure it\'s connected to the Raspberry Pi.')
    sys.exit(1)
  
  print('Loading a model...')
  classifier = vision.Classifier(vision.CLASSIFICATION_MODEL)
  classes = classifier.get_classes(frame)
  if classes:
    print('USB Accelerator okay.')

  print('\nAll tests complete.')

