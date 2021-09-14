from aiy.coral import vision
from pycoral.adapters.detect import BBox
from pycoral.utils.dataset import read_label_file

# BGR (not RGB) colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

# Load the neural network model
detector = vision.Detector(vision.OBJECT_DETECTION_MODEL)
labels = read_label_file(vision.OBJECT_DETECTION_LABELS)

# Define the protected fence region
width, height = vision.VIDEO_SIZE
xmin = 0
ymin = 0
xmax = int(width * 0.5)
ymax = int(height * 0.5)
fence_box = BBox(xmin, ymin, xmax, ymax)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    # Draw the fenced region
    vision.draw_rect(frame, fence_box, color=BLUE, thickness=3)
    
    # Detect only objects with at least 50% confidence
    objects = detector.get_objects(frame, threshold=0.5)
    
    # Review all detected objects and check for
    # objects that look like a person (ignore the rest)
    for obj in objects:
        label = labels.get(obj.id)
        if 'person' in label:            
            # Get size of the person, the fence, and the overlap between them
            person_area = obj.bbox.area
            fence_area = fence_box.area
            overlap_area = BBox.intersect(obj.bbox, fence_box).area
            
            # Check if more than 30% of the person is in the fence,
            # OR if more than 50% of the fence is obscured by the object
            # (such as if the person is very close to the camera)
            if (overlap_area / person_area) > 0.3 or (overlap_area / fence_area) > 0.5:
                # They are in the fence; draw their box red
                vision.draw_rect(frame, obj.bbox, color=RED)
            else:
                # They are outside the fence; draw them green
                vision.draw_rect(frame, obj.bbox, color=GREEN)
