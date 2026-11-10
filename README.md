<div align="center">
  <img width="100%" src="assets/header.gif" alt="Motion Mamba 2.0">
</div>

# Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control

This is the official repository for the paper:

> **Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control**
>
> [Yue Huang](https://github.com/hyue060825-boop)\*, [Tianshan Zhang](https://neptune-t.github.io/)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, [Jayeon Kang](https://github.com/JayeonKangNature), [Akide Liu](https://scholar.google.com/citations?user=1HjSeKgAAAAJ&hl=en), [Sungjoon Choi](https://scholar.google.com/citations?user=1_PHYMEAAAAJ&hl=en), [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ), and [Hao Tang](https://ha0tang.github.io/)<sup>‡</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author.
>
> #### [Paper]() | [Website](https://aigeeksgroup.github.io/MotionMamba2.0/) | [Models](https://huggingface.co/AIGeeksGroup/MotionMamba2.0) | [Data](https://huggingface.co/datasets/AIGeeksGroup/MotionMamba2.0)


This is the clean collaboration and paper-code repository. Keep this repository
focused on reusable code, documented interfaces, and reproducible scripts. Do
not commit local checkpoints, generated videos, W&B runs, caches, or private
server paths.

## Current Layout

```text
ardy/
```

`ardy/`

Main ARDY basecode used as the current humanoid motion generation/control
baseline and visualization stack.

## Branch Layout

Use branches to keep the paper-code repository readable:

- `main`: clean ARDY basecode.
- `mamba-transformer-keneral`: standalone Mamba-side operator code.
- `ardymamba`: planned branch for the cleaned experimental ARDY/Mamba fork
  imported from the server.

## Planned Layout

```text
ardy/                         # main ARDY basecode
docs/                         # paper-code notes, setup, repo map
```

The older direct ARDY Transformer-to-Mamba modification from the server should
be inspected before import. If it is useful, put the cleaned version on the
`ardymamba` branch. Do not import large experiment outputs.
