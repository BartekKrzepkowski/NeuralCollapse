import time

import torch

# the network latency,
# the persistant mode, the GPU power-saving model,
# GPU warm-up
# some dummy examples
# host (CPU) and device (GPU)
# unsynchronized execution
# a low latency network ??
# a Throughput = 1 / Latency = the maximal number of input instances the network can process in time a unit (s)

class TimerCPU:
    # prawdopodobnie mierzy czas niedokładnie, ale ten czas jest zawsze niedoszacowany (. < true time)
    def __init__(self):
        self.logger = None
        self.times = {}
        
    def set_logger(self, logger):
        self.logger = logger

    def start(self, name):
        self.start_time = time.time()
        self.name = name
        
    def stop(self):
        self.end_time = time.time()
        self.durraton = self.end_time - self.start_time
        self.times[f'time/cpu_{self.name}'] = self.durraton
        
    def log(self, global_step):
        self.times[f'time/cpu'] = global_step
        self.logger.log_scalars(self.times, global_step)
        self.times = {}
    
    
class TimerGPU:
    def __init__(self):
        self.logger = None
        self.times = {}
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        
    def set_logger(self, logger):
        self.logger = logger

    def start(self, name):
        self.starter.record()
        self.name = name
        
    def stop(self):
        self.ender.record()
        torch.cuda.synchronize()
        self.durraton = self.starter.elapsed_time(self.ender)
        self.times[f'time/gpu_{self.name}'] = self.durraton
        
    def log(self, global_step):
        self.times[f'time/gpu_{global_step}'] = global_step
        self.logger.log_scalars(self.times, global_step)
        self.times = {}