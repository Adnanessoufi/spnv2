# Preliminary Experiment Results
This directory records the frozen HIL evaluation results obtained during the
initial single-GPU experiments.
Model A and Model B were trained using a physical batch size of 2 on one GPU. This differed substantially from the intended global batch size of 16 and resulted in approximately eight times
more optimizer updates per epoch.
So i decided i will repeat it using different setup
## Training pipelines
- **Official SPNv2:** checkpoint released by the SPNv2 authors.
- **Model A:** ImageNet initialization -> SPEED+ Tango training.
- **Model B:** ImageNet initialization -> SPE3R pretraining -> SPEED+ Tango training.
- **Random seed:** 2021.
- **Evaluation:** frozen testing on SPEED+ Lightbox and Sunlamp.
## Frozen HIL results
| Model | Test domain | Orientation error | Translation error | Final pose score |
|---|---|---:|---:|---:|
| Official SPNv2 | Lightbox | 6.442 deg | 0.175 m | 0.141 |
| Official SPNv2 | Sunlamp | 10.878 deg | 0.225 m | 0.227 |
| Model A | Lightbox | approximately 13.0 deg | 0.392 m | 0.287 |
| Model A | Sunlamp | approximately 19.0 deg | 0.327 m | 0.389 |
| Model B | Lightbox | approximately 15.6 deg | 0.465 m | 0.343 |
| Model B | Sunlamp | approximately 30.3 deg | 0.545 m | 0.618 |
Lower values are better for all three metrics.
## Interpretation
The official SPNv2 checkpoint reproduced strong Lightbox and Sunlamp results
through the same local testing pipeline. This confirmed that the dataset,
labels, evaluation code and metric calculations were functioning correctly.
Model A did not reproduce the official checkpoint performance. The main
difference was the training regime: the preliminary run used a global batch
size of 2 instead of 16 while retaining the original learning rate and epoch
count. Consequently, it performed approximately eight times more optimizer
updates.
Model B performed worse than Model A under this preliminary setup. This may
indicate negative transfer from SPE3R, but no reliable scientific conclusion
should be drawn until both models are retrained under the same corrected
distributed setup.
The frozen evaluation was repeated and produced the same aggregate results,
confirming that the reported scores were reproducible.
## Raw evaluation logs
Primary per-image evaluation logs are retained in:
- `frozen_hil_seed2021/`
- `official_spnv2_frozen/`
## Next experiment
Both Model A and Model B will be retrained using:
- 2 NVIDIA A100 GPUs
- 8 images per GPU
- global batch size 16
- DistributedDataParallel
- synchronized BatchNorm
- identical optimization settings for both models
Model A will be trained and evaluated first to verify that the baseline can be
reproduced before running Model B.

## BatchNorm recalibration diagnostic

The backbone BatchNorm statistics of the preliminary Model A checkpoint were
recalibrated using synthetic Tango training images without updating the learned
model weights.

The Lightbox final pose score improved from approximately 0.287 to 0.257.
However, this remained substantially worse than the official SPNv2 score of
0.141.

This indicated that BatchNorm running statistics contributed to the performance
gap, but they were not the only cause. The learned weights were also affected by
the incorrect small-batch training regime.
