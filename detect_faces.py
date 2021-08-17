from aiy.coral import vision
from pycoral.adapters.detect import BBox

# Load the neural network model
detector = vision.Detector(vision.FACE_DETECTION_MODEL)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
  faces = detector.get_objects(frame, threshold=0.1)
  vision.draw_objects(frame, faces)
