# Multi-Feature Implicit Model (MIMO)

## Setup

---

**Clone this repo**
```
git clone https://github.com/6cyc6/MIMO.git
cd MIMO
```
**Create a new virtual environment (e.g. anaconda environment)**
```
conda create -n mimo python=3.8
conda activate mimo
```

**Install dependencies**

```
pip install numpy cython  # required for building the project
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .
```

## Acknowledgements

---

This code was built upon [Occupancy Networks](https://github.com/autonomousvision/occupancy_networks), [Neural Descriptor Field (NDF)](https://github.com/anthonysimeonov), 
[Neural Interaction Field and Template](https://github.com/zzilch/NIFT) and 
[Relational NDF](https://github.com/anthonysimeonov/relational_ndf). 
Please also check their repos for more details.
