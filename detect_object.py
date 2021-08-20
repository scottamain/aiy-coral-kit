from aiy.coral import vision
from pycoral.utils.dataset import read_label_file

# Load the neural network model
detector = vision.Detector(vision.OBJECT_DETECTION_MODEL)
labels = read_label_file(vision.OBJECT_DETECTION_LABELS)

# Run a loop to get images and process them in real-time  
for frame in vision.get_frames():
    objects = detector.get_objects(frame, threshold=0.4)
    vision.draw_objects(frame, objects, labels)
