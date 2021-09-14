#!/bin/bash
# Copyright 2021 Google Inc.
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

# This script is designed to run on the Raspberry Pi OS.
# It verifies the camera, installs Coral libraries, and clones our projects repo.

set -e

export REPO_PROJECT="scottamain"
export REPO_NAME="aiy-coral-kit"

function get_repo () {
    # Check if repo exists
    if [ -d "${REPO_NAME}" ]; then
      echo "${REPO_NAME} directory exists. Skipping git clone."
      return
    fi
    
    # Check if git is installed
    git=`dpkg -l | grep "ii  git " | wc -l`
    if [ $git -eq 0 ]; then
        echo "Installing git..."
        sudo apt-get update
        apt-get install git -y
    fi

    git clone https://github.com/${REPO_PROJECT}/${REPO_NAME}
}

function enable_camera () {
    CAM=$(sudo raspi-config nonint get_camera)
    if [ $CAM -eq 1 ]; then
        sudo raspi-config nonint do_camera 0
        echo "Camera is now enabled, but you must reboot for it to take effect."
        echo "After reboot, run setup.sh again to finish the setup."
      
        while true; do
          read -p "Reboot now? (y/n) " yn
          case $yn in
            [Yy]* )
              echo "Rebooting..."
              sudo reboot now; break;;
            [Nn]* )
              echo "Setup cancelled. You must reboot to continue setup."; exit;;
            * )
              echo "Please answer yes or no.";;
          esac
        done
    else
      echo "Camera is already enabled."
    fi
}

echo
echo "Checking the camera..."
enable_camera

echo
echo "Installing required packages for Coral..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update --allow-releaseinfo-change

sudo apt-get -y install \
  libedgetpu1-std \
  python3-pycoral \
  python3-tflite-runtime \
  python3-numpy \
  python3-pyaudio \
  python3-opencv \
  zip \
  unzip

while true; do
  echo
  echo "We've installed the standard library for the Coral Edge TPU."
  echo "For increased Edge TPU speeds, you can install the max-performance library."
  read -p "Do you want to install the max-performance library? (y/n) " yn
  case $yn in
    [Yy]* )
      sudo apt-get install libedgetpu1-max -y; break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done
echo "Done."

echo
echo "Downloading AIY Coral Kit repo..."
get_repo
echo "Done."

echo
echo "Setup is complete. You should now verify it all works:"
echo "  1. Be sure your camera is connected to the Raspberry Pi."
echo "  2. Connect the Coral USB Accelerator to the Raspberry Pi."
echo "     If the USB Accelerator is already connected, unplug it and plug it back in."
echo "  3. Download our models and run a test with this command:"
echo "     bash ${REPO_NAME}/run_demo.sh"
echo
