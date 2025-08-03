# SemCovert

## Visualization
The visualizations are available in the [visualization](./visualization)
### UCF101
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/UCF_01.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/UCF_02.mp4" type="video/mp4">
    </video>
  </div>
</div>
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/UCF_03.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/UCF_04.mp4"type="video/mp4">
    </video>
  </div>
</div>

### DAVIS
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/DAVIS_01.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/DAVIS_02.mp4"  type="video/mp4">
    </video>
  </div>
</div>
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/DAVIS_03.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/DAVIS_04.mp4"  type="video/mp4">
    </video>
  </div>
</div>

### MOT17
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/MOT17_01.mp4"  type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/MOT17_02.mp4" type="video/mp4">
    </video>
  </div>
</div>
<div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/MOT17_03.mp4" type="video/mp4">
    </video>
  </div>
  <div style="flex: 1; min-width: 200px; margin: 10px;">
    <video width="100%" controls>
      <source src="./visualization/MOT17_04.mp4" type="video/mp4">
    </video>
  </div>
</div>

## Quick Start

This script provides a quick test for the SemCovert model to embed and reconstruct secret videos.

```bash
python quick_test.py \
  --cover_video_path /path/to/cover_video.mp4 \
  --secret_video_path /path/to/secret_video.mp4 \
  --pre_model_path /path/to/best_model_weights.pth \
  --output_dir ./visual_result \
  --config pre_model_config \
  --batch_size 1 \
  --target_size 256 256 \
  --device cuda
```

## Train
**Coming soon** — This section will be available shortly.

## Reproducibility

All experimental results can be reproduced using the code files provided in the [`Exp`](./Exp) directory. Refer to the individual scripts for specific training, testing, and evaluation procedures.