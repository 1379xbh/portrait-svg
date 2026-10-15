# Informative Drawings inference adapter

`generator.py` retains only Generator / ResidualBlock from the authors' MIT-licensed code:
https://github.com/carolineec/informative-drawings/blob/main/model.py

The original MIT license is included in `LICENSE`.
Changes: removed training-only classes and unused imports, including torchvision.
The network layers and checkpoint names are unchanged.

Weights are obtained from the authors' own demo:
https://huggingface.co/spaces/carolineec/informativedrawings

The demo calls the checkpoints **style 1** (`model.pth`) and **style 2** (`model2.pth`).
We preserve those labels rather than assuming a named art style.
`informative.py` uses RGB values in [0,1], keeps aspect ratio, pads to a multiple of 4,
and crops only padding after inference. Photos are processed locally.
