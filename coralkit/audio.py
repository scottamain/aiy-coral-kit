import numpy as np
import csv

import tflite_runtime.interpreter as tflite

from coralkit import audio_recorder


def class_names(class_map_csv):
    """Read the class name file and return a list of strings."""
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header
        return np.array([display_name for (_, _, display_name) in reader])


def classify_audio(model_file, labels_file, callback,
                   audio_device_index=0, sample_rate_hz=16000,
                   negative_threshold=0.6, num_frames_hop=33):
    """Acquire audio, preprocess, and classify."""
    downsample_factor = 1
    if sample_rate_hz == 48000:
        downsample_factor = 3
    # Most microphones support this
    # Because the model expects 16KHz audio, we downsample 3 fold
    recorder = audio_recorder.AudioRecorder(
        sample_rate_hz,
        downsample_factor=downsample_factor,
        device_index=audio_device_index)
    #  feature_extractor = Uint8LogMelFeatureExtractor(num_frames_hop=num_frames_hop)
    labels = class_names(labels_file)

    interpreter = tflite.Interpreter(model_path=model_file)
    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]['index']
    embeddings_output_index = output_details[1]['index']
    spectrogram_output_index = output_details[2]['index']

    keep_listening = True
    with recorder:
        print("Ready for voice commands...")
        while keep_listening:
            # TODO: Adjust num_audio_frames for voice commands
            waveform, _, _ = recorder.get_audio(num_audio_frames=8**4)
            waveform = waveform / 32768.0  # Convert to [-1.0, +1.0]
            waveform = np.squeeze(waveform.astype('float32'))
            print('length: ', len(waveform))
            # print('waveform: ', waveform)
            interpreter.resize_tensor_input(waveform_input_index,
                                            [len(waveform)],
                                            strict=True)
            interpreter.allocate_tensors()
            interpreter.set_tensor(waveform_input_index, waveform)
            interpreter.invoke()
            scores, embeddings, spectrogram = (
                interpreter.get_tensor(scores_output_index),
                interpreter.get_tensor(embeddings_output_index),
                interpreter.get_tensor(spectrogram_output_index))

            # (N, 521) (N, 1024) (M, 64)
            print(scores.shape, embeddings.shape, spectrogram.shape)

            # Scores is a matrix of (time_frames, num_classes) classifier scores.
            # Average them along time to get an overall classifier output for the clip.
            results = np.mean(scores, axis=0)
            top5_i = np.argsort(results)[::-1][:5]

            # print('\n'.join('  {:12s}: {:.3f}'.format(labels[i], results[i])
            #                for i in top5_i))

            prediction = np.argmax(results)

            keep_listening = callback(labels[prediction], results[prediction])


class WaveformExtractor:

    def __init__(self):
        self._clear_buffers()

    def _clear_buffers(self):
        self._audio_buffer = np.array([], dtype=np.int16).reshape(0, 1)
        self._waveform = np.zeros(3 * 16000, dtype=np.float32)

    def get_audio_sample(recorder, duration):
        print('foo')


class AudioClassifier:
    """Performs classifications with a speech detection model.
    Args:
      model_file: Path to a `.tflite` speech classification model (compiled for the Edge TPU).\
      labels_file: Path to the corresponding labels file for the model.
      audio_device_index: Specify the device card for your mic. Defaults to 0.
        You can check from the command line with `arecord -l`. On Raspberry Pi, your mic must be via
        USB or a sound card HAT, because the Pi's headphone jack does not support mic input.
    """

    def __init__(self, model_file, labels_file, audio_device_index=0):
        self._thread = threading.Thread(target=classify_audio,
                                        args=(
                                            model_file, labels_file,
                                            self._callback,
                                            audio_device_index), daemon=True)
        self._queue = queue.Queue()
        self._thread.start()

    def _callback(self, label, score):
        self._queue.put((label, score))
        return True

    def next(self, block=True):
        """
        Returns a speech classification.
        Each time you call this, it pulls from a queue of recent classifications. So even if there are
        many classifications in a short period of time, this always returns them in the order received.
        Args:
          block (boolean): Whether this function should block until the next classification arrives (if
            there are no queued classification). If False, it always returns immediately and returns
            None if the classification queue is empty.
        """
        try:
            result = self._queue.get(block)
            self._queue.task_done()
            return result
        except queue.Empty:
            return None


VOICE_MODEL = 'models/voice_commands_v0.7_edgetpu.tflite'
VOICE_LABELS = 'models/labels_gc2.raw.txt'
