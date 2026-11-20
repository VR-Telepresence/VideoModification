import queue
import threading
import time

q = queue.Queue()

def consumer():
    while True:
        sent = q.get()
        delay = (time.perf_counter_ns() - sent) / 1e6
        print(f"Delay: {delay:.3f} ms")

threading.Thread(target=consumer, daemon=True).start()

while True:
    q.put(time.perf_counter_ns())
    time.sleep(0.1)