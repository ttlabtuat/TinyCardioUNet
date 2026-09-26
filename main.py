import numpy as np
import torch
from floppy import FLOPpyTracker
from torchinfo import summary
from TinyCardioUNet.model import TinyCardioUNet
from decompose_tinycardiounet_vbmf import decompose_tinycardiounet_vbmf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Initialize the model architecture and load the base weights
model = TinyCardioUNet(in_channels=6)
model.load_state_dict(torch.load("./weights/tinycardiounet_base.pt", map_location=device))

# 2. Decompose the model (VBMF-based tensor decomposition) and load the compressed weights
decompose_tinycardiounet_vbmf(model, verbose=False)
model.load_state_dict(torch.load("./weights/tinycardiounet_compressed_finetuned_ep10.pt", map_location=device))
model = model.to(device)
model.eval()

# 3. Prepare a dummy IMU signal
# Shape: (batch_size, channels, 256Hz*4s samples)
# A batch of 1 sequence, IMU 6-channels, 1024 samples.
dummy_imu = np.random.rand(1, 6, 1024).astype(np.float32)
dummy_imu = torch.from_numpy(dummy_imu).to(device=device)

# 4. Predict ECG signal
with torch.no_grad():
    predicted_ecg = model(dummy_imu)  # (1, 1, 1024)

# 5. Count parameters (torchinfo) and FLOPs (floppy)
stats = summary(model, input_size=(1, 6, 1024), device=device, verbose=0)

tracker = FLOPpyTracker(run_name="TinyCardioUNet")
tracker.run(model=model)
with torch.no_grad():
    model(dummy_imu)
tracker.stop()
flops = tracker.report().model_forward_flop
