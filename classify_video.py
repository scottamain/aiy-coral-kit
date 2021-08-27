import argparse
from pycoral.utils.dataset import read_label_file
from aiy.coral import vision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', default=vision.CLASSIFICATION_MODEL,
                  help='File path of .tflite file. Default is vision.CLASSIFICATION_MODEL')
parser.add_argument('-l', '--labels', default=vision.CLASSIFICATION_LABELS,
                  help='File path of labels file. Default is vision.CLASSIFICATION_LABELS')
args = parser.parse_args()

# Load the neural network model
classifier = vision.Classifier(args.model)
labels = read_label_file(args.labels)

# Run a loop to get images and process them in real-time
for frame in vision.get_frames():
    # Get only the top result, when confidence score is >= 30%
    classes = classifier.get_classes(frame, top_k=1, threshold=0.3)
    if classes:
        score = classes[0].score
        label = labels.get(classes[0].id)
        vision.draw_label(frame, label)
