# rl-inter-spar


## Installation

### Create environment and install packages
```
conda create -n spar python==3.10
conda activate spar
conda install cmake ffmpeg
python -m pip install -r requirements.txt
AutoROM --accept-license
```
### Install submodules
```
git submodule update --init --recursive
```

### Playground
`playground.py` needs a `ffmpeg` installation to save the video.

