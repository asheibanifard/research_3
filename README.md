# SIREN — Sinusoidal Representation Networks

Trains a SIREN model on volumetric TIFF data.

## Usage

```bash
python src/siren.py \
    --input path/to/volume.tif \
    --model_dir logs/siren \
    --hidden_features 256 \
    --hidden_layers 5 \
    --omega_0 30.0 \
    --lr 1e-4 \
    --epochs 1000 \
    --batch_size 65536 \
    --steps_til_summary 500 \
    --epochs_til_checkpoint 20 \
    --early_stopping_patience 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Path to input TIFF file |
| `--model_dir` | `logs/siren` | Base output directory; each run writes to `<model_dir>/s<layers>_<neurons>/` |
| `--hidden_features` | 256 | Width of each hidden layer |
| `--hidden_layers` | 5 | Number of hidden layers |
| `--omega_0` | 30.0 | Frequency scaling factor |
| `--lr` | 1e-4 | Adam learning rate |
| `--epochs` | 1000 | Number of training epochs |
| `--batch_size` | 1048576 | Voxels sampled per step |
| `--steps_til_summary` | 20 | Steps between checkpoint saves and TensorBoard writes |
| `--epochs_til_checkpoint` | 20 | Epochs between epoch-level checkpoints |
| `--early_stopping_patience` | 50 | Epochs with no improvement before stopping |
| `--use_kernel` | off | Use fused CUDA sine kernel (requires compiling `siren_cuda.cu`) |
| `--no_amp` | off | Disable automatic mixed precision |
| `--compile` | off | Enable `torch.compile` (requires working Triton/Inductor) |

## 3DGS Config Workflow

`src/3dgs.py` now supports a YAML-first workflow for reproducible runs:

```bash
python src/3dgs.py --config configs/3dgs.yml
```

CLI flags still win over YAML values, so you can keep a stable base config and
override only the fields you want to change for a run:

```bash
python src/3dgs.py --config configs/3dgs.yml --n_init 20000 --out logs/3dgs/ablation_a
python src/3dgs.py --config configs/3dgs.yml --no_use_kernel
```

The config loader expects a YAML mapping and rejects unknown keys. `--volume`
must be provided either in the YAML file or on the CLI.

Minimal example:

```yaml
volume: 10-2900-control-cell-05.oif-C0.v3dpbd.tif
swc_path: 10-2900-control-cell-05.oif-C0.v3dpbd.swc
out: logs/3dgs/run
device: cuda

n_init: 10000
max_gaussians: 50000
init_scale: 0.05
init_inten: 0.1

epochs: 500
steps_per_epoch: 50
batch: 2048
chunk_n: 1024

densify_from_step: 500
densify_until_step: 15000
densify_interval: 200

lr_means: 1.6e-4
lr_means_final: 1.6e-6
lr_scales: 5.0e-3
lr_quats: 1.0e-3
lr_inten: 1.0e-2

swc_init: true
ckpt_interval: 2000
```

## Changelog

### 2026-04-22

#### Fix: half-precision (float16) error with fused CUDA sine kernel

`AT_DISPATCH_FLOATING_TYPES` in `siren_cuda.cu` only handles `float` and `double`.
When AMP (`use_amp=True`) is active, activations are cast to `float16`, causing:

```
RuntimeError: "sine_forward" not implemented for 'Half'
```

**Fix** — `_SineFn.forward` and `_SineFn.backward` now cast the input to `float32`
before calling the CUDA kernel and cast the result back to the original dtype:

```python
# before
return _load_cuda_kernel().forward(x, omega_0)

# after
return _load_cuda_kernel().forward(x.float(), omega_0).to(x.dtype)
```

The same upcast/downcast is applied to `fused_sine` for the non-kernel path:

```python
# before
return torch.sin(omega_0 * x)

# after
return torch.sin((omega_0 * x).float()).to(x.dtype)
```

#### Fix: slow training due to excessive disk I/O

Three sources of per-step overhead:

1. `torch.save(model.state_dict())` was called every `steps_til_summary` steps
   (default 20), producing thousands of checkpoint writes during a full run.
2. Three `writer.add_scalar` calls fired on every step regardless of
   `steps_til_summary`, flooding TensorBoard with writes.
3. The per-step loss scalar (`train_loss.item()`) was computed and logged
   redundantly in two places.

**Fix** — TensorBoard writes, checkpoint saves, and console logging are now all
gated behind the `do_summary` flag (i.e. only every `steps_til_summary` steps).
The backward pass is also moved before `.item()` so the GPU is not stalled
waiting on a CPU sync before gradients are computed.

**Recommended**: increase `--steps_til_summary` from the default 20 to 500+,
and reduce `--batch_size` from `2**20` (1 M) to `2**16` (65 K) to avoid
memory-bandwidth saturation with large intermediate tensors.

#### Feature: per-architecture output directories

Checkpoints and TensorBoard summaries are now written to a subdirectory named
after the model architecture, so runs with different configurations do not
overwrite each other.

**Directory layout:**

```
<model_dir>/
  s<hidden_layers>_<hidden_features>/
    checkpoints/
    summaries/
```

**Example** — `--model_dir logs/siren --hidden_layers 5 --hidden_features 256`
produces `logs/siren/s5_256/checkpoints/` and `logs/siren/s5_256/summaries/`.
