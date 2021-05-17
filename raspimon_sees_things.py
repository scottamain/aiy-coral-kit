from pycoral.utils.dataset import read_label_file
import vision

# Main program ------------------------

# Load the neural network model
classifier = vision.Classifier(vision.CLASSIFICATION_MODEL)
labels = read_label_file(vision.CLASSIFICATION_LABELS)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
  # Get list of all recognized objects in the frame
  classes = classifier.get_classes(frame)
  label_id = classes[0].id
  score = classes[0].score
  label = labels.get(label_id)
  print(label, score)
  # Draw the label name on the video
  vision.draw_classes(frame, classes, labels)
