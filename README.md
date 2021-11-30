# AIY Maker Kit API and examples

The `aiymakerkit` Python APIs in this repo greatly simplify the amount of code
required to perform common operations with TensorFlow Lite models, such as
running inference with image classification, object detection, and pose
estimation models. It also includes scripts to collect training
images and perform transfer learning with an image classification model.

This project is designed specifically for a Raspberry Pi with a Coral USB
Accelerator and a camera (though it may be repurposed for other systems as
well).

For detailed documentation using the Raspberry Pi, including the `aiymakerkit`
API reference, see https://g.co/coral/kit-guide.

## Install

If you're on a Raspberry Pi, follow the setup guide at
https://g.co/coral/kit-guide.

For other manual installs, **you must first install the `libedgetpu` and 
`pycoral` libraries**. And we assume you are using the Coral USB Accelerator,
so follow [the instructions here](https://coral.ai/docs/accelerator/get-started/).

Then clone this repo and install as follows:

```
git clone https://github.com/google-coral/aiy-maker-kit.git

cd aiymakerkit

python3 -m pip install .
```
