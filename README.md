# Accelerate LLM inference
Simple and efficient pytorch-native transformer text generation based on gpt-fast.

Featuring:
1. Very low latency
2. <1000 lines of python
3. No dependencies other than PyTorch and sentencepiece
4. int8/int4 quantization
5. Speculative decoding (We add support for llama-68m, llama-160m)
6. Prefill caching (We build it)


## Installation
[Download PyTorch nightly](https://pytorch.org/get-started/locally/)
Install sentencepiece and huggingface_hub
```bash
pip install sentencepiece huggingface_hub
```

To download llama models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.
Then login with `huggingface-cli login`



## Downloading Weights
Models tested/supported
```text
JackFram/llama-68m
JackFram/llama-160m
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-70b-chat-hf
```

For example, to convert Llama-2-7b-chat-hf
```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_REPO
```

## Generate Text

Model definition in `model.py`, generation code in `generate.py`.

```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "Hello, my name is"
```

To squeeze out a little bit more performance, you can also compile the prefill with `--compile_prefill`. This will increase compilation times though.

## Quantization
Choose device to use by
```bash
# The current support devices: cuda, cpu
export DEVICE=cuda
```
### Int8 Weight-Only Quantization
To generate this version of the model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int8.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int8
```
To run with int8, just pass the int8 checkpoint to generate.py.
```bash
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model_int8.pth --device $DEVICE
```

### Int4 Weight-Only Quantization
To generate int4 version of model
```bash
# Spits out model at checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth
python quantize.py --checkpoint_path checkpoints/$MODEL_REPO/model.pth --mode int4 --groupsize 32 --device $DEVICE
```

To run with int4, just pass the int4 checkpoint to generate.py.
```bash
python generate.py --checkpoint_path checkpoints/$MODEL_REPO/model_int4.g32.$DEVICE.pth --compile --device $DEVICE
```

## Speculative Sampling
To generate with speculative sampling (DRAFT_MODEL_REPO should point to a smaller model compared with MODEL_REPO).

```
export DRAFT_MODEL_REPO=JackFram/llama-68m
python generate.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth
```

## Prefill Caching
To generate with prefill caching
```
python generate_prefill.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth
```

