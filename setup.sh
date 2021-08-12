#!/bin/bash
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

echo
echo "Enabling the camera..."
sudo raspi-config nonint do_camera 0
echo "Done."

echo
echo "Installing required packages for Coral..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
#echo libedgetp1-max libedgetpu/accepted-eula boolean true | debconf-set-selections

sudo apt-get update && sudo apt-get -y install \
  libedgetpu1-max \
  python3-pycoral \
  python3-tflite-runtime \
  python3-numpy \
  python3-pyaudio \
  python3-opencv
echo "Done."

echo
echo "Downloading AIY Coral Kit repo..."
get_repo
echo "Done."

echo
echo "Downloading model files..."
(cd "${REPO_NAME}" && bash install_requirements.sh)
echo "Done."

echo
echo "Setup is done."

while true; do
  echo
  read -p "Run a quick hardware test? (y/n) " yn
  case $yn in
    [Yy]* )
      echo
      (cd "${REPO_NAME}" && python3 test.py); break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done


while true; do
  echo
  read -p "Try the face detection demo? (y/n) " yn
  case $yn in
    [Yy]* )
      echo "Press Q to quit the demo."
      (cd "${REPO_NAME}" && python3 detect_faces.py); break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done

echo
echo "All done! See more demos in /${REPO_NAME}."