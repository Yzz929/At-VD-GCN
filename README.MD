# At-VD-GCN:Trajectory Forecasting with Graph Convolutional Neural Net-works Based on Prior-Aware and Information Fusion
### Yzz <br />

### At-VD-GCN
<div align='center'>
<img src="images.png"></img>
</div>
<br />

We propose a pedestrian trajectory prediction model based on prior-aware and information fusion. To make the input information more effective, for the different importance of the input trajectory information, we design the time information weighting module to weigh the observed trajectory information differently at different moments based on the original observed trajectory infor-mation. To reduce the impact of redundant information on trajectory prediction and to improve the interaction between pedestrians, we present a spatial interaction module of multi-pedestrians and a topological graph fusion module. In addition, we use a temporal convolutional network module to obtain the temporal interactions between pedestrians. Compared to Social-STGCNN, the ex-perimental results show that the model we proposed reduces the average displacement error (ADE) and final displacement error (FDE) by 32% and 38% on the datasets of ETH and UCY, respectively. Moreover, based on this model, we design an autonomous driving obstacle avoidance system that can ensure the safety of road pedestrians effectively.

Our model inference speed is 0.0014s/frame (714Hz) using only 6.16K parameters.
### Citaion
You can cite our paper using: 
```
@inproceedings{mohamed2020social,
  title={At-VD-GCN:Trajectory Forecasting with Graph Convolutional Neural Net-works Based on Prior-Aware and Information Fusion},
  author={Yzz},
  booktitle={--},
  pages={--},
  year={2023}
}
```


#### Model
<div align='center'>
<img src="images.png"></img>
</div>
<br />

#### Predictions sample

<div align='center'>
<img src="images/social-stgcnn-pred.gif"></img>
</div>
<br />

### Setup: 
The code was written using python 3.6. 
The following libraries are the minimal to run the code: 
```python
import pytorch
import networkx
import numpy
import tqdm
```
or you can have everything set up by running: 
```bash
pip install -r requirements.txt
```
### Using the code:
To use the pretrained models at `checkpoint/` and evaluate the models performance run:
```bash
test.py
```

To train a model for each data set with the best configuration as in the paper, simply run:
```bash
./train.sh  
```
Please note: The initiation of training and testing might take a while as the code creates a cache to store the spatio-temporal graphs.
<br />
Exact CVPR version: https://github.com/abduallahmohamed/Social-STGCNN/tree/ebd57aaf34d84763825d05cf9d4eff738d8c96bb

Check our work in the 3D mode: https://github.com/abduallahmohamed/Skeleton-Graph
