#!/bin/bash
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

# This script installs everything you need to use the aiymakerkit library
# and run the example code in examples/.
#
# Just navigate to this directory in your terminal and run the script:
#
#    bash run_demo.sh
#
# For more instructions, see https://aiyprojects.withgoogle.com/maker/

set -e

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash ${SCRIPT_DIR}/examples/install_requirements.sh

if [[ ! $DISPLAY ]]; then
  echo "ERROR: No display detected. This demo requires a desktop display."
  exit 1;
fi

while true; do
  echo
  read -p "Run a quick hardware test? (y/n) " yn
  case $yn in
    [Yy]* )
      echo
      if !(cd "${SCRIPT_DIR}" && python3 "scripts/test_cam.py"); then
        exit 1;
      fi
      break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done

while true; do
  echo
  read -p "Try the face detection demo? (y/n) " yn
  case $yn in
    [Yy]* )
      (cd "${SCRIPT_DIR}" && python3 "examples/detect_faces.py"); break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done

echo
echo "You can try more code in ${SCRIPT_DIR}/examples/."