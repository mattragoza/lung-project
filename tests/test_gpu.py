import torch
print('CUDA available: ', torch.cuda.is_available())
print('CUDA version:   ', torch.version.cuda)
print('Device count:   ', torch.cuda.device_count())
print('Device name:    ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No device found')
print('CUDNN enabled:  ', torch.backends.cudnn.enabled)

