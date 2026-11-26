import torch


class Gpu:
    def __init__(self, i):
        self.stream = torch.cuda.Stream(device=i)


class GPUManager:
    """ Responsible for managing multiple GPUs and distributing free or ideal gpu"""
    def __init__(self):
        self.gpus:list[Gpu] = list()
        self.__setup_gpus()
        self.__parallax_counter = 0
        
    def __setup_gpus(self):
        if len(self.gpus)!= 0:
            raise Exception("Gpu streams already setup")
        for i in range(torch.cuda.device_count()):
            self.gpus.append(Gpu(i))

    def __gpu_is_available(self, gpu:Gpu):
        return gpu.stream.query()

    def __gpu_for_parallax(self):
        self.__parallax_counter+=1
        return self.gpus[self.__parallax_counter%len(self.gpus)]

    def get_stream(self):
        return self.__gpu_for_parallax().stream






