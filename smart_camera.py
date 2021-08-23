import os.path
import cv2
from datetime import datetime
from time import time
from aiy.coral import vision
from pycoral.adapters.detect import BBox

PICTURE_DIR = '/home/pi/Pictures'
DELAY_SECS = 3
snap_time = 0

def box_is_in_box(bbox_a, bbox_b):
    if (bbox_a.xmin > bbox_b.xmin and bbox_a.xmax < bbox_b.xmax) and (
        bbox_a.ymin > bbox_b.ymin and bbox_a.ymax < bbox_b.ymax):
        return True
    return False

# Load the neural network model
detector = vision.Detector(vision.FACE_DETECTION_MODEL)

# Define the auto shutter detection zone
width, height = vision.VIDEO_SIZE
xmin = int(width * 0.25)
xmax = int(width - (width * 0.25))
ymin = int(height * 0.2)
ymax = int(height - (height * 0.2))
camera_bbox = BBox(xmin, ymin, xmax, ymax)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
  faces = detector.get_objects(frame)
  
  # Count the number of faces in the detection zone
  faces_in_box = 0
  for face in faces:
      if box_is_in_box(face.bbox, camera_bbox):
          faces_in_box += 1
  
  # If everyone's in the box and some time has passed since last photo
  if faces and faces_in_box == len(faces) and time() - snap_time > DELAY_SECS:
      timestamp = datetime.now()
      filename = "SMART_CAM_" + timestamp.strftime("%Y%m%d_%H%M%S") + '.png'
      filename = os.path.join(PICTURE_DIR, filename)
      vision.save_frame(filename, frame)
      snap_time = time()
      print(filename)
  else:
      # Draw bounding boxes on the faces
      vision.draw_objects(frame, faces)
      # Draw the auto shutter box
      vision.draw_rect(frame, camera_bbox)
