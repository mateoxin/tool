from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False
        # Fix: Use time.perf_counter() instead of deprecated tqdm._time()
        self.last_time = time.perf_counter()

    def pause(self):
        if not self.paused:
            self.paused = True
            # Fix: Use time.perf_counter() instead of deprecated tqdm._time()
            self.last_time = time.perf_counter()

    def unpause(self):
        if self.paused:
            self.paused = False
            # Fix: Use time.perf_counter() instead of deprecated tqdm._time()
            cur_t = time.perf_counter()
            self.start_t += cur_t - self.last_time
            self.last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
