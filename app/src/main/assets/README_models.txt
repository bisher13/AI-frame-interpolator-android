Place your interpolation model files here.

Supported options in this app:
- ONNX Runtime (preferred): rife_fp16.onnx
  - Expected input: two frames concatenated into 6 channels (NCHW or NHWC), output: single mid-frame (3 channels)
  - File name expected by default: rife_fp16.onnx

- TensorFlow Lite: rife_fp16.tflite
  - Expected input (placeholder logic): two frames concatenated + alpha scalar; you may need to adapt FrameInterpolator.tfliteInterpolate to your model's IO.
  - File name expected by default: rife_fp16.tflite

Tip: You can change the file names passed to FrameInterpolator.initialize() in MainActivity.
