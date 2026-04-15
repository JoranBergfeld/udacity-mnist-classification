# MNIST Handwritten Digit Classification

## Introduction
This repository holds a project meant to classify handwritten digits. The fictional case is the same one that originally drove MNIST's creation in the 1990s: automatically reading handwritten ZIP codes on mail so the post office doesn't have to do it by hand. The task here is to build a model that can take a 28×28 grayscale image of a digit and output which digit it is, with enough accuracy that you'd actually trust it on real mail. 

This readme will give you context, objectives and goals which I have for this project. I'll also cover the architecture of the project, and how to run it yourself.

Note that this project is entirely done for education purposes. 

## Context, Objectives and Goals
For this project I wanted to build on what I learned from the CIFAR-10 project and apply the same "run a lot of configurations, analyse the results" approach to a simpler dataset. MNIST is significantly easier than CIFAR-10, which means I could push on the pipeline itself (config files, sweep runner, resumable runs) without the training time becoming prohibitive. I wanted to play around with the following axes: neural network architecture, different optimizers (Adam, AdamW, SGD with momentum), different image augmentations (affine, random erasing, combinations), different learning rate schedulers (cosine annealing, one-cycle), and different regularization presets (dropout, weight decay, label smoothing). I'll explain each of these concerns below, with a reasoning why I would want to try it out.

The rubric for this project only requires 90% test accuracy. I have found that bar to be pretty low, as a 2-epoch Multi-Layer Perceptron cleared it in my iniial testing. I wanted to use the easy rubric as headroom to explore how much each knob actually matters when you're already in the 98–99% range.

### Neural network architectures
The architectures I would like to explore are the following 4:

1. An Multi-Layer Perceptron baseline with two hidden layers. This is the simplest thing that satisfies the rubric: "at least two hidden layers, output softmax over 10 classes".
2. A simple CNN with two convolutional blocks. 
3. A deeper CNN with three convolutional blocks, batch normalization, and dropout. 
4. A small ResNet with residual connections, adapted for 28×28 grayscale. This gave me an excuse to read how people adapt ResNet to small grayscale images. This was also the obvious winner in the CIFAR10 dataset.

#### Hypothesis
I expected the CNNs to beat the Multi-Layer Perceptron meaningfully, because convolutions have the right inductive bias for images — translation equivariance, local receptive fields. The Multi-Layer Perceptron has to learn all of that from scratch. I expected the deeper CNN and the small ResNet to perform roughly similarly and both beat the simple CNN. My guess was that the gap between "best CNN" and "Multi-Layer Perceptron" would be somewhere between 1 and 2 percentage point.

### Image augmentation
Coming off the CIFAR-10 project, I had some intuition around augmentation. MNIST is small, simple, and already pretty clean, so I expected augmentation to matter less here than it did on CIFAR-10. I chose three pipelines:

1. **None** — just `ToTensor` and `Normalize`.
2. **Affine** — small random rotation and translation. Digits in the wild are not perfectly centered, and a 7 rotated 5 degrees is still a 7.
3. **Affine + random erasing** — the affine transform plus randomly masking out small patches of the image after normalization. The theory being that random erasing forces the model to not rely on any single pixel region.

#### Hypothesis
I expected `affine` to be the sweet spot. MNIST digits really do appear slightly rotated and off-center in the raw data, so that augmentation feels natural. I was less sure about random erasing, especially on a 28×28 image, a 4×4 erased patch removes a meaningful chunk of the digit, which seemed like it might be too aggressive. I hypothesized that `affine_erasing` would help Multi-Layer Perceptrons as they have more to generalize over, but might slightly hurt the CNNs, which already handle spatial variation well.

### Schedulers — dynamic learning rate
I compared cosine annealing against one-cycle. Both follow the same general idea from the [cyclical learning rates paper](https://arxiv.org/abs/1506.01186): start with a moderate learning rate, change it over time, and refine at the end. Cosine anneals smoothly from the initial learning rate down to zero. One-cycle starts low, ramps up to a peak, then anneals past the initial rate down to almost zero.

#### Hypothesis
Coming off CIFAR-10, where cosine annealing was a clear win, I expected cosine to win here too. My instinct was that one-cycle would be harder to tune well because it needs a sensible `max_lr`, and I wasn't sure I'd pick a good value without experimentation. I expected AdamW + cosine to be the best combination overall.

### Optimizers
I chose three optimizers to compare: Adam, AdamW, and SGD with Nesterov momentum. Adam and AdamW are nearly identical in code but AdamW decouples weight decay from the gradient update, which matters when you actually use weight decay. SGD is the old-school baseline that everything else was invented to beat. I used the same learning rate (1e-3) across all three, which is the same slightly-unfair setup I had in the CIFAR-10 project. 1e-3 is a perfectly good default for Adam and AdamW, but on the low side for SGD, which typically wants something closer to 0.05–0.1.

#### Hypothesis

I expected Adam and AdamW to perform nearly identically at 1e-3 when no weight decay is applied, and for AdamW to pull ahead slightly when weight decay is active. I expected SGD to lose on average because of the learning rate mismatch, but probably not by as much as you'd see on a harder dataset as MNIST seems forgiving.

### Regularization
I packaged dropout rate, weight decay, and label smoothing into two presets rather than sweeping each independently, because the previous 200+ runs were quite a lot. I ended up with these configurations:
- **Light**: dropout=0.2, weight_decay=0, label_smoothing=0
- **Heavy**: dropout=0.3, weight_decay=5e-4, label_smoothing=0.1

#### Hypothesis
I expected heavy regularization to help the bigger models such as DeeperCNN or SmallResNet and either do nothing or slightly hurt the Multi-Layer Perceptron, which has less capacity to overfit in the first place. Label smoothing is a softer version of a hard target, which makes more sense when there's label ambiguity and MNIST has some: A handful of digits are legitimately hard to read.

## Approach
Same as the CIFAR-10 project: I did not consider the notebook to start with. I wanted to start out with implementation of the above goals within python first, run a bunch of tests and analyse the results. The approach I wanted to take is have an easy command line interface in which I could invoke the entire pipeline for a specific configuration with the enhancement that configurations are defined in YAML files rather than passed as CLI flags, so the sweeps themselves are version-controllable artifacts.

The reasoning behind this is that I wanted to be able to run 48 training configurations unattended, or though the night. Having all the logic in a notebook would have meant running cells manually for each configuration. With a proper command line interface and YAML-defined sweeps, I can kick off all the training before going to bed and have results ready in the morning. The notebook still exists for visualization and the Udacity submission, but the heavy lifting happens in the scripts and configs.

The module structure mirrors the sibling CIFAR-10 project almost exactly as I used it to start this project off and not have to google how to do some simple things again. You'll find the same file names, same function naming conventions, same flat artifact layout. The enhancement is the YAML configuration layer on top and a handful of training-loop features that the CIFAR project didn't need: a validation split with best-checkpoint tracking, early stopping, label smoothing, and CUDA, MPS or CPU runtime detection so the same code runs on my desktop and my laptop.

## Project Structure

```
mnist/                               # Python package with all the logic
    cli.py                           # Command line interface, sweep expansion, run scheduling
    data.py                          # Data loading, transforms, val split, notebook helpers
    augmentation.py                  # Augmentation strategies (none, affine, erasing, affine_erasing, randaugment)
    models.py                        # Multi-Layer Perceptron, SimpleCNN, DeeperCNN, SmallResNet
    optim.py                         # Optimizer and scheduler factories
    train.py                         # Training loop, device detection, label-smoothed NLL
    evaluate.py                      # Test-set evaluation with per-class accuracy
    analysis.py                      # Confusion matrix, per-class report, misclassified samples
    save.py                          # Model save/load, metrics persistence, run-name derivation
    config.py                        # Dataclass schema + YAML load/dump
configs/                             # Hand-authored YAML configs + sweep definitions
    baseline_Multi-Layer Perceptron.yaml
    cnn_basic.yaml
    cnn_deep_affine_cosine.yaml
    resnet_small.yaml
    sweep_overnight.yaml             # 48-run grid, grid + exclude rules
docs/                                # Udacity rubric, project instructions
results/                             # Per-run metrics as JSON (generated, gitignored)
models/                              # Saved model weights (generated, gitignored)
data/                                # Downloaded MNIST dataset (generated, gitignored)
MNIST_Handwritten_Digits-STARTER.ipynb  # Notebook for Udacity submission
```

## How to run this project?

### Requirements
This project uses `uv` for dependency management. You'll need Python 3.10+ and `uv` installed. To set up:

```bash
uv sync --extra dev
```

`[tool.uv.sources]` pins `torch` / `torchvision` to the `pytorch-cu124` index on Windows and Linux, so `uv sync` installs CUDA 12.4 wheels by default. On macOS you get the default PyPI wheels, which ship with MPS support built in. The pipeline's `get_device()` picks CUDA over MPS over CPU at runtime, so the same code runs unchanged across machines.

For faster training, the command line interface dynamically determines if the GPU can be leveraged, and will do so if CUDA is detected. This speeds up training drastically. I would *HIGHLY* recommend that you train on CUDA if you have it, but CPU training is supported. On my machine (RTX 4060 Ti), the full 48 configurations took about 3 hours 15 minutes. On CPU this would take overnight or longer.

### Notebook approach
1. Define and run kernel
2. Walk through each cell. The notebook's first cell runs `pip install -r requirements.txt`, which is kept in sync with `pyproject.toml` via `uv export --format requirements-txt --no-hashes --no-dev -o requirements.txt`.

### Code first approach
The command line interface supports YAML configs, YAML sweeps, and CIFAR-style ad-hoc CLI flags (for quick one-off experiments). Results are saved as JSON files in `results/` and model weights in `models/`.

```bash
# Run a single configuration from a YAML file
uv run mnist --config configs/baseline_Multi-Layer Perceptron.yaml

# Run the full 48-configuration overnight sweep
uv run mnist --sweep configs/sweep_overnight.yaml

# Preview what a sweep will run without training anything
uv run mnist --sweep configs/sweep_overnight.yaml --dry-run

# Ad-hoc grid from CLI flags, CIFAR-style
uv run mnist --models Multi-Layer Perceptron simple_cnn --augmentations none affine \
             --optimizers adam --schedulers cosine --epochs 10

# Print a ranked summary across every run in results/
uv run mnist --summary --top 20 --out results/summary.md
```

Run names are derived deterministically from the resolved config, so rerunning the same sweep skips already-completed entries. You can kill the process mid-sweep and resume later, anything already saved to `results/{run_name}.json` is skipped automatically.

## Results
In total I ran **48 training configurations**: a 4×3×2×2 grid over architecture x augmentation x optimizer + scheduler x regularization preset, at 15 epochs each. The full ranked table is in [results/summary.md](results/summary.md).

The highlights:
1. **ResNet is, once again, the clear winner**: `resnet_small` with `affine` augmentation, AdamW + cosine annealing, and heavy regularization hit **99.72%** test accuracy. That's 28 errors out of 10,000 test images. The runner-up, `resnet_small` with light regularization, was at 99.70%, and the top 16 runs are all above 99.5%, dominated by `resnet_small` and `deeper_cnn` architectures. Going in I assumed ResNet would edge out DeeperCNN by noise; it actually won pretty cleanly.

2. **Architecture matters more than anything else**: Multi-Layer Perceptron tops out at 98.58%, SimpleCNN at 99.58%, DeeperCNN at 99.67%, SmallResNet at 99.72%. Each step up the architecture ladder is about a 1% improvement in accuracy, but that's a 40–50% reduction in *error*. The gap between Multi-Layer Perceptron and ResNet is 28 errors vs 142 errors per 10,000 images. You really feel that difference if you're actually reading mail.

3. **`affine` is the sweet spot for CNNs, not `affine_erasing`**: I expected random erasing to add something on top of affine. It didn't. All three CNN architectures hit their best score with plain `affine` augmentation; `affine_erasing` was slightly worse. On a 28x28 image a 4x4 erased patch is enough to hide the entire loop of a 0 or 9, and apparently the CNNs would rather see the full digit.

4. **For Multi-Layer Perceptrons, `affine_erasing` was actually the best augmentation**: the Multi-Layer Perceptron topped out at 98.58% with `affine_erasing`, compared to 98.44% with plain `affine`. The pattern is the opposite of what I saw on CNNs: the Multi-Layer Perceptron benefits from the extra generalization pressure because it has less inductive bias to fall back on.

5. **My SGD + one-cycle concern was mostly wrong**: halfway through the sweep, all I had were Multi-Layer Perceptron results, and SGD + one-cycle was losing to AdamW + cosine on every single one. I got worried that my `max_lr` choice was too low. In the final results across all 48 runs, SGD + one-cycle was actually in 5 of the top 16 slots, only narrowly behind AdamW + cosine. I still think the `max_lr` was underpowered but it turns out MNIST is forgiving enough that it didn't really matter. Lesson learned: don't extrapolate from the Multi-Layer Perceptron rows; the architecture you're looking at has opinions.

6. **Light vs heavy regularization is essentially a coin flip**: The top two runs are the same architecture + augmentation + optimizer + scheduler. Only the regularization preset differs, and they're 0.02% apart. Across the full grid, heavy regularization wins slightly more often than it loses, but not by a margin I'd put any weight on. With 15 epochs and MNIST this is just not the bottleneck.

7. **9, 4, and 7 are the hardest digits**: The per-class F1 scores are consistent across the top runs. 9 is the lowest (+-0.995), followed by 4 and 7. Visually, those are exactly the three digits that blur into each other when handwritten. A 4 with a closed top looks like a 9, and a 7 with a curl looks like a 4. Even the best model can't get around certain images just being genuinely ambiguous. You could argue that humans may make the same amount of mistakes with recognition of numbers.