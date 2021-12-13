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
"""
Setup script to install the aiymakerkit library as a package.

If you plan to modify the aiymakerkit APIs, you should install the package
to be editable:

python3 -m pip install -e .

Then you can make changes to the aiymakerkit Python files and those changes
are instantly available to other programs that import aiymakerkit.
"""

import os
import re
from setuptools import setup


def read(filename):
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def find_version(text):
    match = re.search(r"^__version__\s*=\s*['\"](.*)['\"]\s*$", text,
                      re.MULTILINE)
    return match.group(1)


setup(
    name='aiymakerkit',
    description='Simple API for ML inferencing with PyCoral and TF Lite',
    long_description=read('README.md'),
    license='Apache 2',
    version=find_version(read('aiymakerkit/__init__.py')),
    author='Coral',
    author_email='coral-support@google.com',
    url='https://github.com/google-coral/aiy-maker-kit',
    packages=['aiymakerkit'],
    install_requires=['pycoral>=2.0.0'],
)
