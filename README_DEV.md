

## Installing dependencies
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install gradio , opencv-python, random-fourier-features-pytorch, pytorch360convert, ultralytics
```

## Start Program
```bash
python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_256x256.pth
```
or
```bash
python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_256x256.pth --detector yolo
```



## Latency & Frame Rate Testing
- tested using a RTX 4070S & AMD Ryzen 7 7800x3D
- Reading cam : 0.002s
- 360->viewport: 0.01
- tranformations -> parallax -> transformation: 0.037



### Without multithreading & Multiprocessing 
- FPS: 16-19 -> dropof of 11-14 frames per second
- Latency (creation of parallax): 0.04 secs (in python alone)

### WIth Multithreading
- FPS (in python backend) without Unity: 27-30
- FPS (in python backend) with Unity: 19-20
- Question: why does the performance sink so drastically? Needs investigation. 