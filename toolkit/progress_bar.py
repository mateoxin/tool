from tqdm import tqdm
import time


class ToolkitProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paused = False
        # Fix: Use time.perf_counter() instead of deprecated tqdm._time()
        self.last_time = time.perf_counter()
        # Ensure start_t exists for compatibility with newer tqdm versions
        if not hasattr(self, 'start_t'):
            self.start_t = time.perf_counter()
        if not hasattr(self, 'last_print_t'):
            self.last_print_t = time.perf_counter()

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
            # Safe access to tqdm internal attributes
            if hasattr(self, 'start_t'):
                self.start_t += cur_t - self.last_time
            elif hasattr(self, '_start_t'):
                self._start_t += cur_t - self.last_time
            if hasattr(self, 'last_print_t'):
                self.last_print_t = cur_t
            elif hasattr(self, '_last_print_t'):
                self._last_print_t = cur_t

    def update(self, *args, **kwargs):
        if not self.paused:
            super().update(*args, **kwargs)
