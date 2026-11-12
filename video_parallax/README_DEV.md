

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

### Start performance monitoring
While the main program is running, execute this in another terminal:
```bash
python video_parallax/performance_monitoring.py
```



## Performance drop in combination with Unity (probably tue to a lack of computational resources)

#### Problem description
During performance testing something interesting seems to happen:

Once Unity is started the framerate(FPS) drops considerably by around 10fps and the latency increases. Both are bad signs. 

If the imshow window is focused again the framerate increases to the previous value again and the latency is lowered (even with the unity scene still running).

If the program has no imshow window, the performance drop persists with no way to increase it though focus. Only shutting the unity scene down works.

Normal:

        ╔════════════════════════════════════════════╗
        ║             PERFORMANCE MONITOR            ║
        ╠══════════════╦════════════╦════════════════╣
        ║ Stage        ║ FPS        ║ Latency (ms)   ║
        ╠══════════════╬════════════╬════════════════╣
        ║ Reading      ║      24.00 ║           4.03 ║
        ║ Parallax     ║      24.00 ║          36.92 ║
        ║ Postproc     ║      24.00 ║           0.00 ║
        ║ Sending      ║      24.00 ║           0.35 ║
        ╠══════════════╩════════════╬════════════════╣
        ║ TOTAL LATENCY             ║          41.30 ║
        ╚═══════════════════════════╩════════════════╝
        ╔══════════════╦════════════╦════════════════╗
        ║ Object detect║       1.00 ║         769.12 ║
        ╚══════════════╩════════════╩════════════════╝
        

Lowered:

        ╔════════════════════════════════════════════╗
        ║             PERFORMANCE MONITOR            ║
        ╠══════════════╦════════════╦════════════════╣
        ║ Stage        ║ FPS        ║ Latency (ms)   ║
        ╠══════════════╬════════════╬════════════════╣
        ║ Reading      ║      15.00 ║           4.32 ║
        ║ Parallax     ║      16.00 ║          56.23 ║
        ║ Postproc     ║      16.00 ║           0.15 ║
        ║ Sending      ║      16.00 ║           0.42 ║
        ╠══════════════╩════════════╬════════════════╣
        ║ TOTAL LATENCY             ║          61.12 ║
        ╚═══════════════════════════╩════════════════╝
        ╔══════════════╦════════════╦════════════════╗
        ║ Object detect║       0.00 ║        1005.07 ║
        ╚══════════════╩════════════╩════════════════╝


#### Likely cause

The likely cause of that is that Unity is simply consuming less ressources when the imshow window is focused. 

Normal performance:

![img_1.png](img_1.png)

Lowered Performance:

![img.png](img.png)


#### Conclusion
As of now it seems to be a problem entirely related to the hardware performance with no software fix available. 

