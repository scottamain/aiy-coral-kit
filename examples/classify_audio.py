from coralkit import audio


def callback(label, score):
    print('CALLBACK: ', label, score)
    if label.startswith('exit'):
        return False  # stop listening
    return True  # keep listening


audio.classify_audio(model_file='models/yamnet.tflite',
                     labels_file='models/yamnet_class_map.csv',
                     callback=callback)
