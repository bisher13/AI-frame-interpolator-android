# Video Interpolation Android App

An Android application that performs video interpolation to create smooth slow-motion videos by generating intermediate frames between existing frames.

## Features

- 📹 **Video Frame Extraction**: Automatically extracts frames from selected videos
- 🤖 **AI Frame Interpolation**: Uses machine learning to generate smooth intermediate frames
- ⚡ **Adjustable Frame Rate**: Choose from 2x to 10x frame rate multiplier
- 📱 **Modern UI**: Clean Material Design interface
- 🎬 **Video Encoding**: Automatically encodes interpolated frames back into video

## Technical Stack

- **Language**: Kotlin
- **ML Framework**: TensorFlow Lite
- **Media Processing**: MediaCodec, MediaMuxer, MediaMetadataRetriever
- **UI**: Material Design Components, ViewBinding
- **Async**: Coroutines

## Project Structure

```
app/
├── src/main/
│   ├── java/com/videointerpolation/app/
│   │   ├── MainActivity.kt              # Main UI and orchestration
│   │   ├── ml/
│   │   │   └── FrameInterpolator.kt     # ML-based frame interpolation
│   │   └── utils/
│   │       ├── VideoProcessor.kt        # Frame extraction
│   │       └── VideoEncoder.kt          # Video encoding
│   ├── res/
│   │   ├── layout/
│   │   │   └── activity_main.xml        # Main UI layout
│   │   └── values/
│   │       ├── strings.xml
│   │       ├── colors.xml
│   │       └── themes.xml
│   └── AndroidManifest.xml
└── build.gradle
```

## How It Works

1. **Frame Extraction**: The app extracts individual frames from the selected video
2. **Interpolation**: For each pair of consecutive frames, the ML model generates intermediate frames
3. **Encoding**: All frames (original + interpolated) are encoded into a new high-framerate video
4. **Output**: The resulting video has a higher frame rate, creating smooth slow-motion effect

## Setup Instructions

### Prerequisites

- Android Studio Arctic Fox or newer
- Android SDK 24 or higher
- Device or emulator running Android 7.0+

### Build Steps

1. Open the project in Android Studio
2. Sync Gradle files
3. Build the project: `Build > Make Project`
4. Run on device or emulator

### Adding ML Model (Optional Enhancement)

The current implementation uses linear interpolation. For better results, add a TensorFlow Lite model:

1. Download a frame interpolation model (e.g., FILM, RIFE)
2. Convert to TFLite format if needed
3. Place the `.tflite` file in `app/src/main/assets/`
4. Update `FrameInterpolator.kt` to load and use the model

Recommended models:
- **FILM** (Frame Interpolation for Large Motion)
- **RIFE** (Real-Time Intermediate Flow Estimation)
- **DAIN** (Depth-Aware Video Frame Interpolation)

## Usage

1. Launch the app
2. Tap "Select Video" to choose a video from your device
3. Adjust the frame rate multiplier (2x-10x) using the slider
4. Tap "Process Video" to start interpolation
5. Wait for processing to complete
6. Find the output video in the app's files directory

## Permissions

The app requires the following permissions:
- `READ_EXTERNAL_STORAGE` / `READ_MEDIA_VIDEO`: To access videos
- `WRITE_EXTERNAL_STORAGE`: To save processed videos (Android 9 and below)

## Performance Considerations

- **Processing Time**: Varies based on video length, resolution, and device performance
- **Memory Usage**: Large videos may require significant memory
- **Storage**: Processed videos are saved to external storage
- **Battery**: Video processing is CPU/GPU intensive

## Future Enhancements

- [ ] Add GPU acceleration for faster processing
- [ ] Implement advanced ML models (FILM, RIFE)
- [ ] Add video preview before processing
- [ ] Support batch processing
- [ ] Add quality presets (fast/balanced/high quality)
- [ ] Implement background processing with notifications
- [ ] Add video trimming/cropping before interpolation

## Troubleshooting

**Issue**: Video encoding fails
- **Solution**: Ensure sufficient storage space and valid video format

**Issue**: Out of memory errors
- **Solution**: Try processing shorter videos or reduce resolution

**Issue**: Slow processing
- **Solution**: Use lower frame rate multiplier or shorter videos

## License

This project is provided as-is for educational purposes.

## Credits

Built with Android Studio and TensorFlow Lite
