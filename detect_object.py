from aiy.coral import vision
from pycoral.adapters.detect import BBox
from pycoral.utils.dataset import read_label_file

RED = (0, 0, 255) # BGR (not RGB)

def get_fence(img_dims):
    img_width, img_height = img_dims
    xmin = 0
    ymin = 0
    xmax = int(img_width * 0.5)
    ymax = int(img_height * 0.5)
    return BBox(xmin, ymin, xmax, ymax)
    

# Main program ------------------------

# Load the neural network model
detector = vision.Detector(vision.OBJECT_DETECTION_MODEL)
labels = read_label_file(vision.OBJECT_DETECTION_LABELS)

# Create fence bounding-box relative to the video size
fence_box = get_fence(vision.VIDEO_SIZE)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    objects = detector.get_objects(frame, threshold=0.5)
    
    # Draw the fenced region
    vision.draw_rect(frame, fence_box, thickness=3)
    
    # Draw bounding boxes for detected objects of interest
    for obj in objects:
        label = labels.get(obj.id)
        if 'person' in label:
            vision.draw_rect(frame, obj.bbox, color=RED)
            
            # Get size of the object, the fence, and overlap between them
            object_area = obj.bbox.area
            fence_area = fence_box.area
            overlap_area = BBox.intersect(obj.bbox, fence_box).area
            
            # If more than 30% of object is in the fence,
            # OR if more than 50% of the fence is obscured by the object
            if (overlap_area / object_area) > 0.3 or (overlap_area / fence_area) > 0.5:
                vision.draw_rect(frame, fence_box, color=RED)
