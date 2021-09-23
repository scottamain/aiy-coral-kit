
Image classification
--------------------

.. autoclass:: coralkit.vision.Classifier
    :members:


Object detection
----------------

.. autoclass:: coralkit.vision.Detector
    :members:


Pose detection
--------------

.. autoclass:: coralkit.vision.PoseDetector
    :members:
    :undoc-members:

.. autofunction:: coralkit.vision.get_keypoint_types

.. autoclass:: coralkit.vision.KeypointType
    :members:
    :undoc-members:


Pose classification
-------------------

.. autoclass:: coralkit.vision.PoseClassifier
    :members:
    :undoc-members:


Camera & drawing
----------------

.. autofunction:: coralkit.vision.get_frames

.. autofunction:: coralkit.vision.save_frame

.. automodule:: coralkit.vision
    :members: draw_classes, draw_objects, draw_pose, draw_label, draw_rect, draw_circle
    :member-order: bysource


.. Until we can put objects.inv on coral.ai
.. |Class| replace:: ``Class``
.. _Class: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.classify.Class
.. |Object| replace:: ``Object``
.. _Object: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.detect.Object
.. |BBox| replace:: ``BBox``
.. _BBox: https://coral.ai/docs/reference/py/pycoral.adapters/#pycoral.adapters.detect.BBox