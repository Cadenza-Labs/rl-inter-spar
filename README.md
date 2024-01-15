# rl-inter-spar


## Installation

### submodules
```
git submodule update --init --recursive
```

### AutoROM
Run:
```
AutoROM --accept-license
```

### Create environment and install packages
```
conda create -n spar python==3.10
conda activate spar
conda install cmake ffmpeg
pip install -r requirements.txt
AutoROM
```

### Playground
`playground.py` needs a `ffmpeg` installation to save the video.


