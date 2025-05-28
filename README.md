# VPR-RIR

This project contains the code for the paper "Unified Variational and Physics-aware Model for Room Impulse Response
Estimation" by Louis Lalay, Matthieu Fontaine and Roland Badeau. If this code is useful for your research, please cite the [paper](https://link.html):

```
@inproceedings{lalay2023vpr,
  title={Unified Variational and Physics-aware Model for Room Impulse Response Estimation},
  author={Lalay, Louis and Fontaine, Matthieu and Badeau, Roland},
  booktitle={2025, IEEE Interspeech},
  year={2025},
  organization={IEEE}
}
```

# Summary
VPR-RIR is a physically based model for **room impulse response (RIR) estimation**. Knowing the dry source and the reverberant signal, the model estimates the RIR even in noisy conditions. The model is based on a variational approach and uses 2 filters to estimate the RIR parameters. The parameters are estimated using a gradient descent algorithm.

## Repository structure
1. **data/** contains the csv files of the audios used as source, noise and RIR.
2. **runs/** contains the results. In each folder, there are all the necessary files to reproduce the results:
   1. The config file
   2. The orgiginal RIR, the dry source, the noise and the reverberant signal ias wav files
   3. The state dict of the model, including the estimated RIR
   4. The tensorboard logs
3. **baseline.py** contains the baselines used in the paper.
4. **data.py** contains the pyTorch wrapper for wav files.
5. **main.py** contains the main function to run the model.
6. **physical_model.py** contains the physical model used in the paper.
7. **vprrir.py** contains the main class for the VPR-RIR model.
8. The other files are dependencies for the different modules.