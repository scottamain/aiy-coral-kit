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

from aiymakerkit import audio

import argparse
import models

def callback(label, score):
    print('CALLBACK: ', label, '=>', score)
    return True  # keep listening


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_device_index', type=int, default=0)
    args = parser.parse_args()

    audio.classify_audio(model_file=models.YAMNET_MODEL,
                         labels_file=models.YAMNET_LABELS,
                         callback=callback,
                         audio_device_index=args.audio_device_index)
