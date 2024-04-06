import torch
import GPUtil


def get_device():
    if torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    return device

def schedule_gpu():
    if torch.cuda.is_available():
        id = GPUtil.getFirstAvailable(order="memory")[0]
        torch.cuda.set_device(id)