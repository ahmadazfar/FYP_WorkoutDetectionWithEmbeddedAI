# FYP_WorkoutDetectionWithEmbeddedAI

ai8x-training
WDDataset.py - Network Loader (Load sensor measurements) 
wd-model.py - Train data using pytorch 

Inference
main.c - Used for inference:
Measure MPU6050 dataAI Model Training (via ai8x-training)
WDDataset.py – Dataset loader that reads sensor measurements (e.g., accelerometer & gyroscope data).

wd-model.py – Defines and trains a PyTorch model to classify workout movements.

🚀 Inference (Deployment)
main.c – Embedded inference implementation:

Collects real-time data from MPU6050.

Loads trained model weights.

Runs inference directly on-device.
Use trained weights for inference
