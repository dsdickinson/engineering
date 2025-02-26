#!/usr/bin/env python3

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# nvcc --version
# echo $CUDA_HOME
# echo $LD_LIBRARY_PATH

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
print("Torch Version: {}".format(torch.__version__))
print("CUDA Available?: {}".format(torch.cuda.is_available()))
print("Device count: {}".format(torch.cuda.device_count()))
print("Device name: {}".format(torch.cuda.get_device_name(0)))
print("Device properties: {}".format(torch.cuda.get_device_properties(0)))
print("Arch list: {}".format(torch.cuda.get_arch_list()))

#torch.cuda.set_per_process_memory_fraction(0.5) # Limit to 50%
#torch.cuda.empty_cache()
