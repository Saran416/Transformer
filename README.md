# Transformer from Scratch

This repository contains a PyTorch implementation of a Transformer model, inspired by the original paper [_Attention Is All You Need_ (2017)](https://arxiv.org/abs/1706.03762). The model is trained from scratch on an English-to-German dataset using only basic PyTorch components, with custom implementations of multi-head attention, positional encoding, and encoder-decoder architecture.

## Features

- Encoder-decoder Transformer (no external libraries)
- Custom attention, positional encoding, and training loop
- Character-level tokenization
- Outputs valid German sequences

## Project Structure

```
src/
├── model/
├── data/
notebooks/
outputs/
```
