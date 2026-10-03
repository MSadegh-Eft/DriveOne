# Code Walkthrough: `smoke_test_pe.py`

Source: [`scripts/smoke_test_pe.py`](../../scripts/smoke_test_pe.py)

This script checks the released Perception Encoder interface before a fusion model is designed.

## Arguments and result object

The script accepts a PE config name, optional image, text string, device, `--pretrained`, output path, and `--strict`. It initializes a JSON-compatible result with status `not_ready`, requested config, and warnings.

The current implementation accepts `--text` for future interface work, but does not tokenize or run the text path yet. That is an important limitation: today it is a vision-tower smoke test only.

## Optional imports

It imports PyTorch, Pillow, and the official `core.vision_encoder.pe` and `transforms` modules inside a `try` block. If dependencies are unavailable, it writes a structured error. Non-strict mode exits successfully after reporting the missing dependency; strict mode exits `2`.

This behavior makes the script usable in a clean checkout while preserving a hard failure for a claimed PE verification.

## Model and input construction

`VisionTransformer.from_config` loads the requested config and, when requested, the official pretrained checkpoint. The image transform is taken from the official package using the model’s image size. With no image, a zero tensor is created so only interface shapes can be inspected; its values are explicitly marked synthetic.

## Three forward paths

The script measures:

1. `model.forward_features(image, strip_cls_token=False)` — the sequence including the class token when the implementation provides it;
2. `model.forward_features(image, strip_cls_token=True)` — the sequence with the class token removed, used as the candidate patch-token candidate;
3. `model(image)` — the model’s pooled output path.

It records tensor shapes, image size, context length if exposed, parameter dtype, device, and separate latency measurements. CUDA synchronization is used around timing when the device string begins with `cuda`.

## Error and output handling

Model or forward errors produce status `failed` and a warning. The JSON is printed and optionally written to the requested output. Strict mode returns `2` unless the status is `ok`.

## What this script proves and does not prove

It can establish whether the installed official checkpoint/config exposes a usable tensor sequence and pooled output. It does not prove that the sequence is semantically grounded, that patch tokens are better than pooled features, that text conditioning works, or that the model fits deployment latency. It also does not yet measure candidate crop consistency or fusion-transformer compute.
