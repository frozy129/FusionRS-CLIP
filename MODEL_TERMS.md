# Model and Data Terms

## Code

The Python code in this repository is released under the MIT License in
`LICENSE`.

## Model weights

`model/model.safetensors` is a fine-tuned derivative of
`openai/clip-vit-base-patch32`. The FusionRS fine-tuned weight modifications
are released under CC BY-NC 4.0 for research and non-commercial use. Users
must also comply with the terms and model card of the upstream OpenAI CLIP
model.

Upstream model:
https://huggingface.co/openai/clip-vit-base-patch32

Upstream code license:
https://github.com/openai/CLIP/blob/main/LICENSE

## Training data

This model package does not redistribute the 600K source RGB images or all
translated image bytes. FusionRS combines sources with different terms.
Dataset release must follow the source-aware, index-only policy documented in
the paper. Users are responsible for obtaining upstream data and complying
with each source license.

## Intended use

The model is intended for research on remote-sensing RGB, infrared-style, and
sensor-captured infrared representation learning. It is not a calibrated
thermal measurement system and must not be used as the sole basis for
surveillance, targeting, emergency response, safety-critical decisions, or
public policy.

