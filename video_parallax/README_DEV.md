

## Installing dependencies
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install gradio , opencv-python, random-fourier-features-pytorch, pytorch360convert, ultralytics, ndi-python, cv2_enumerate_cameras 
```

## Start Program
```bash
python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_256x256.pth
```
or
```bash
python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_256x256.pth --detector yolo
```

python main.py  --checkpoint .\checkpoint\checkpoint_blender_pvsnet_512x512.pth --detector yolo



### Start performance monitoring
While the main program is running, execute this in another terminal:
```bash
python video_parallax/performance_monitoring.py
```

# Optimisations
## Using workers instead of linear processing
Linear processing caused the fps to drop dastically.
Imagine: You have 30fps input. So every second 30 frames need to be processed. 
That is about 33ms per frame. 
So you have 33ms until your camera receives the next frame. 
If you take longer (what happened with our relatively good gpu), the next frame has already passed before you finish with your current frame. Then you have to wait until you get the next frame. That essentially reduces the fps by half. 

Solution 1: Store the frames using thread that only accepts the frames and do all the calulcatoins for the frame in another thread linearly. 

Problem of Solution 1: If there are cpu processes that take time (initially the viewport extraction), the gpu has times, where it is idle -> Solution 2

Solution 2: But the GPU process in its own worker that only handles GPU tasks. 
If the worker is done with its current task it gets the next one from a queue. So, the GPU is constantly processing, thus increasing the fps throughput. 

Next problem: Because the parallax creation takes longer than 33ms on average, the next parallax creation will have to wait a few ms until the previous is done. 
For the second frame that is just 2ms, for the next one it is +2ms = 4ms, for the next one +2ms = 6ms. 
So, latency builds up


## Multiple streams per gpu
It was observed that the latency of the viewport extraction ON THE GPU took far longer than antifipated and a bit longer than on the cpu: 35ms, about as long as the parallax creation. 
We tried to reduce the time needed for the viewport extraction to reduce the total latency.

We figured out that the problem was, that the viewport extraction needed to wait. 
So we tried to use 2 streams, one that is reposnible for viewport extraction and another for parallax creation.
Later however, we found out that the actual problem is the same as explained in the end of [Using workers instead of linear processing](#Using-workers-instead-of-linear-processing)
The gpu is simply busy with other things and needs time until it can handle the viewport extraction: 

### Here the testing report: 
It was tested and the latency for the preprocessing went down. 
However, the total latency and especially the max latency went up. 

**Why the total latency was not reduced**: The bottleneck is the parallax. After the preprocessing is done, the thread still needs to wait until the parallax is done to proceed. The preprocessing was pushed forward but it cannot preceed from there, making multiple streams per gpu completely useless.

**Why the total latency even went up**: Probably because the streams interrupted each others work through reservation of ressources and memory transfer between the streams.  Maybe there is even more to it, like inefficient sheduling of tasks to the two streams. But the problems remains the same so it does not really matter.

**Conclusion**: It is nice for the performance monitoring but bad for the performance. The GPU only takes about 3ms for the viewport extraction, so it can simply be done directly after the previous parallax creation is finished. 
That only delays the next parallax creation by 3ms (if done on the gpu instead of the cpu) which is much better compared to the latency introduced by 2 streams. 
Again, there is no gain from using 2 streams. 
The latency could be reduced by another 3ms, if the CPU is faster than 30ms with the viewport extraction so that the GPU can skip that and directly go over to the parallax creation. 

## Using multiple gpus
Why: Urently the gpu takes more than 


Idea 1: Using one additional gpu (that can be weaker than the main gpu) for viewport creation: max gain =  about 3ms faster

Idea 2: Using 2 gpus that handle the entire frame processing pipeline. Max gain = 35ms (time needed for parallax creation) through prevention of latency buildup (explained in previous problems).  
-> 35-70ms to consistently 35 ms (with another gpu as good as the first one)
The problem with this solution is, that the latency improvednt is strongly coupled to the gpu speed of the second gpu. Any ms it takes longer than the other is additional latency for the entire pipeline.

Optimisation example for Idea 2: the second gpu takes 40ms -> Make the first gpu take images until it reaches the 40ms latency (through delay builtup), then let the second gpu take an image. That should reset the latency built-up on the first gpu. 


Still under test

# Performance measurements
## Preproc Latency explained
The preprocessing latency is shown as 3-35ms. However it only really takes 3ms. Is appears like it because 



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

TODO:
- refactor worker-> create superclass
- make postproc worker to prevent images from getting lost due to too long preprocessing time. 
- test other library for viewpoiunt extraction