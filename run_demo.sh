#!/bin/bash
set -e

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash ${SCRIPT_DIR}/install_requirements.sh

if [[ ! $DISPLAY ]]; then
  echo "ERROR: No display detected. This demo requires a desktop display."
  exit;
fi

while true; do
  echo
  read -p "Run a quick hardware test? (y/n) " yn
  case $yn in
    [Yy]* )
      echo
      if !(cd "${SCRIPT_DIR}" && python3 test.py); then
        exit;
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
      (cd "${SCRIPT_DIR}" && python3 "detect_faces.py"); break;;
    [Nn]* ) break;;
    * ) echo "Please answer yes or no.";;
  esac
done

echo
echo "See more demos in ${SCRIPT_DIR}/."