# Audio Affect Probe

Audio Affect Probe is a small reproducible research project for dynamic music affect estimation on DEAM.

The project began with a broader synesthesia idea. The original question was whether a brain inspired fast path, slow path, and learned salience gate could produce a useful internal affect state before any color mapping was attempted. The experiments produced a clear answer. Temporal modeling helped, but the proposed salience mechanism did not survive fresh seed validation. A later separate head experiment also failed to produce a reliable gain.

The final repository keeps the full experiment ladder instead of hiding negative results. The goal is reproducibility and honest model comparison, not a novelty claim.

## Final status

The retained reference model is `dual`.

It combines a short context convolution path with a recurrent context path, then learns a shared fusion before predicting valence and arousal.

The `salience` model remains in the repository because it was the main rejected hypothesis. The `dual_split` and `affect_split` models remain because they were used to test whether valence and arousal needed separate output or fusion rules.

The color engine was removed from the final project. There was no defensible objective target for a universal song color, and asking users to provide personal color labels would have made the system depend on information it was supposed to infer.

## Research questions

1. Does temporal context improve dynamic valence and arousal estimation over a frame MLP baseline?
2. Does adding a short context path improve the slow recurrent model?
3. Does a learned gate provide a stable advantage over an ungated dual path model?
4. Does the learned gate actually react to measurable acoustic or affective change events?
5. Does separating valence and arousal heads produce a stable gain?

## Dataset

The project uses DEAM audio and dynamic valence and arousal annotations.

Use the averaged dynamic annotations per song. The preparation script expects one arousal CSV, one valence CSV, and the extracted audio directory.

The preprocessing pipeline does the following:

* Recursively finds audio files by song ID.
* Reads the annotated part of each song.
* Converts audio to a 96 bin log mel representation.
* Stores one cache file per song.
* Creates a deterministic 80/10/10 train, validation, and test split at song level.
* Creates 30 second training chunks after the song split.
* Keeps an effective batch size of 16 with gradient accumulation on small GPUs.

The dataset itself is not included in this repository.

Official DEAM page:

<https://cvml.unige.ch/databases/DEAM/>

## Model variants

### baseline_mlp

Shared log mel projection followed by a frame MLP prediction head.

Parameters in the recorded run: 29,442.

### slow

Shared projection followed by a two layer GRU and the shared prediction head.

Parameters in the recorded run: 227,842.

### dual

Shared projection followed by two temporal paths.

The fast path uses dilated temporal convolution.

The slow path uses a two layer GRU.

The two states are concatenated and passed through learned fusion before the shared prediction head.

Parameters in the recorded runs: 313,090.

### salience

The same fast and slow paths are used, but the fast state is added to the slow state through a learned sigmoid gate. The gate also receives spectral flux and absolute energy change cues.

Parameters in the recorded runs: 313,475.

This model was rejected as the final model because its advantage was not stable across fresh seeds, and telemetry did not show a reliable relation between the gate and the intended salience events.

### dual_split

The ordinary dual representation is kept, but valence and arousal use separate scalar heads.

Parameters in the recorded fresh seed runs: 329,602.

This model was also rejected as the final model because its four seed mean CCC was effectively equal to the simpler dual model.

### affect_split

Valence uses ordinary dual fusion. Arousal uses the gated fast residual over the slow state. The encoders are shared and the output heads are separate.

This model was tested as a targeted follow up and did not beat `dual_split` on the first controlled probe.

## Metrics

The primary metric is Concordance Correlation Coefficient, reported as CCC.

The trainer also records Pearson correlation and RMSE for valence and arousal.

Mean CCC is the arithmetic mean of valence CCC and arousal CCC.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

### Git Bash, Linux, or WSL

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

For CUDA, install a PyTorch build that matches the CUDA environment on the machine.

Verify the device:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Smoke test

```bash
python scripts/smoke_test.py
```

Expected final line:

```text
SMOKE_TEST_OK
```

## Prepare DEAM

Git Bash example:

```bash
python scripts/prepare_deam.py \
  --audio-dir "C:/path/to/DEAM/audio" \
  --arousal-csv "C:/path/to/DEAM/annotations/arousal.csv" \
  --valence-csv "C:/path/to/DEAM/annotations/valence.csv" \
  --out data/processed
```

Single line form:

```bash
python scripts/prepare_deam.py --audio-dir "C:/path/to/DEAM/audio" --arousal-csv "C:/path/to/DEAM/annotations/arousal.csv" --valence-csv "C:/path/to/DEAM/annotations/valence.csv" --out data/processed
```

A quick preprocessing test can be done with 20 songs:

```bash
python scripts/prepare_deam.py --audio-dir "C:/path/to/DEAM/audio" --arousal-csv "C:/path/to/DEAM/annotations/arousal.csv" --valence-csv "C:/path/to/DEAM/annotations/valence.csv" --out data/processed_test --limit 20
```

## Small GPU setting

The included configs use 30 second chunks, batch size 4, and gradient accumulation 4. The effective batch size is 16.

For a 4 GB GPU, Git Bash can also use:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If memory is still insufficient, change the YAML values to:

```yaml
batch_size: 2
gradient_accumulation_steps: 8
```

The effective batch size remains 16.

## Reproduction protocol

The experiment history is intentionally reproducible in stages.

### Stage 1. Seed 42 architecture ablation

Run all four original models:

```bash
bash scripts/train_all.sh cuda
```

Plain Windows PowerShell:

```bash
python -m audio_affect_probe.train --config configs/baseline_mlp.yaml --device cuda --seed 42 --zip-results
python -m audio_affect_probe.train --config configs/slow.yaml --device cuda --seed 42 --zip-results
python -m audio_affect_probe.train --config configs/dual.yaml --device cuda --seed 42 --zip-results
python -m audio_affect_probe.train --config configs/salience.yaml --device cuda --seed 42 --zip-results
```

Recorded test results:

```text
model          valence CCC    arousal CCC    mean CCC    valence RMSE    arousal RMSE
baseline_mlp      0.154035       0.626143      0.390089      0.311716        0.260645
slow              0.189651       0.686563      0.438107      0.292089        0.233336
dual              0.206242       0.681184      0.443713      0.281869        0.239435
salience          0.197788       0.744911      0.471350      0.305824        0.215179
```

This first run made the salience model look promising, especially for arousal. The next stage tested whether that result survived fresh seeds.

### Stage 2. Fresh seed dual and salience comparison

```bash
python scripts/run_gate_sweep.py --device cuda --seeds 1337 2026 31415
```

Recorded results:

```text
seed     dual val    dual ar    dual mean    salience val    salience ar    salience mean
1337     0.328571    0.720420    0.524496       0.253080       0.729475        0.491277
2026     0.274560    0.756645    0.515602       0.181972       0.714756        0.448364
31415    0.264684    0.701286    0.482985       0.265783       0.746649        0.506216
```

Paired salience minus dual deltas:

```text
seed     arousal CCC delta    valence CCC delta    mean CCC delta
1337          0.009055             -0.075491          -0.033218
2026         -0.041889             -0.092587          -0.067238
31415         0.045363              0.001099           0.023231
```

Across these three fresh seeds, salience had only a 0.004176 mean arousal CCC advantage, but a negative 0.055660 mean valence CCC difference and a negative 0.025742 mean CCC difference.

Including seed 42 gives the following four seed summary:

```text
model       valence CCC mean    arousal CCC mean    mean CCC mean
Dual             0.268514            0.714884          0.491699
Salience         0.224656            0.733948          0.479302
```

The salience model improved arousal on three of four seeds, but the mean gain was small and the valence cost was larger. It was not retained.

### Stage 2b. Gate telemetry

An existing salience checkpoint can be evaluated without training:

```bash
python scripts/eval_checkpoint.py --checkpoint "results/salience_20260830_135332/best.pt" --device cuda
```

The fresh seed sweep also records telemetry automatically for salience runs.

Recorded telemetry:

```text
seed     gate mean    gate and flux r    gate and energy change r    gate and abs arousal change r    event ratio
42        0.132321        -0.011425              -0.065896                    0.025140              1.031826
1337      0.124113         0.048295               0.096499                    0.057605              1.118460
2026      0.111266        -0.137475              -0.106273                   -0.043425              0.897970
31415     0.186265        -0.143199              -0.071569                   -0.009887              1.002397
```

The event ratio compares mean gate value in the top 10 percent of absolute arousal changes against all other moments.

The gate did not show a stable relation to spectral flux, energy change, or absolute arousal change. This means it should not be described as a validated salience mechanism.

### Stage 3. Separate head probe

The next question was whether valence and arousal were interfering through one shared output head.

Run the controlled seed 42 probe:

```bash
python scripts/run_head_probe.py --device cuda --seed 42
```

Recorded results:

```text
model          valence CCC    arousal CCC    mean CCC
dual_split        0.282929       0.737934      0.510432
affect_split      0.242616       0.743589      0.493103
```

The seed 42 result made `dual_split` look useful, so it was tested on fresh seeds before accepting the change.

### Stage 4. Fresh seed dual_split confirmation

```bash
python scripts/run_head_sweep.py --device cuda --seeds 1337 2026 31415
```

Recorded results:

```text
seed     valence CCC    arousal CCC    mean CCC
1337        0.303987       0.736075      0.520031
2026        0.194776       0.679208      0.436992
31415       0.260344       0.729504      0.494924
```

Four seed comparison against the matching dual runs:

```text
model         valence CCC mean    arousal CCC mean    mean CCC mean
Dual               0.268514            0.714884          0.491699
Dual Split         0.260509            0.720680          0.490595
```

Paired dual_split minus dual mean differences across four seeds:

```text
valence CCC difference    -0.008005
arousal CCC difference     0.005797
mean CCC difference       -0.001104
```

The separate heads did not produce a reliable overall improvement. The simpler `dual` model was retained.

## Full automatic reproduction

After DEAM has been prepared, the complete experiment sequence can be launched with:

```bash
python scripts/run_reference_protocol.py --device cuda
```

This runs Stage 1, Stage 2, Stage 3, and Stage 4 in order.

It is intentionally expensive compared with a single run because the point is to reproduce the decisions, not only the final checkpoint.

## Verify the included reference results

The machine readable result tables used in this README are stored under `reference_results`.

Verify their aggregate calculations without training:

```bash
python scripts/verify_reference_results.py
```

## Result package format

Every training run creates a directory under `results` with:

```text
best.pt
best_val_metrics.json
test_metrics.json
summary.json
history.csv
training_curve.png
test_predictions.npz
config.yaml
git.json
```

Salience related runs can also contain:

```text
gate_telemetry.json
test_gate_telemetry.npz
```

With `--zip-results`, the run is also copied to `exports` as one ZIP file.

An existing run can be packaged later:

```bash
python scripts/package_results.py results/your_run_directory
```

## Final interpretation

The strongest stable observation in this experiment series is simple.

Temporal modeling improved arousal estimation substantially over the frame MLP baseline. The slow recurrent model raised seed 42 arousal CCC from 0.626143 to 0.686563. The dual path model improved valence and produced a compact shared temporal representation, but its seed 42 arousal result was close to the slow model.

The learned gate produced a strong arousal result on seed 42, but fresh seed tests showed that the effect was not stable enough to justify a salience claim. Gate telemetry also failed to show a reliable relationship with the acoustic and affective events it was intended to represent.

Separate valence and arousal heads produced a strong result on seed 42, but the advantage disappeared across fresh seeds.

For this reason the final retained architecture is the simpler `dual` model.

## What this project does not claim

* It does not claim biological realism.
* It does not claim that the gate models an amygdala or any other brain structure.
* It does not claim state of the art performance on DEAM.
* It does not claim to infer a listener's personal emotional response.
* It does not claim that music has an objective color.

## Why the original color direction was removed

A music color system needs a defensible target. A universal color target was not available, and a personal adapter would require users to provide the very associations the system was expected to discover.

The project therefore stops where the measurements are meaningful. It estimates dynamic valence and arousal and documents which architectural ideas did and did not survive controlled tests.

## Repository structure

```text
AudioAffectProbe
  configs
    baseline_mlp.yaml
    slow.yaml
    dual.yaml
    salience.yaml
    dual_split.yaml
    affect_split.yaml
  data
  reference_results
  scripts
    prepare_deam.py
    smoke_test.py
    train_all.sh
    run_gate_sweep.py
    eval_checkpoint.py
    run_head_probe.py
    run_head_sweep.py
    run_reference_protocol.py
    verify_reference_results.py
    package_results.py
  src
    audio_affect_probe
      audio.py
      data.py
      metrics.py
      models.py
      train.py
      utils.py
  tests
    test_models.py
```

## Notes on reproducibility

The DEAM split is created during preprocessing and stored in `manifest.json`. Fresh training seeds change model initialization, data loader order, and other training randomness, but they do not create a new song split.

For strict comparison, use the same prepared cache for every model and seed.

Recorded numbers in this repository came from the same prepared DEAM cache and the configs included here.
