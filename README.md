# Pytorch0.4.1_WatchNet: Efficient and Depth-based Network for People Detection in Video Surveillance Systems

# Test using the trained model

```
python pose_detect.py models/posenet.pth
```

<div align="center">
<img src="https://github.com/Qidian213/WatchNET/blob/master/images/0643.png" width="800" height="600">
&nbsp;
<img src="https://github.com/Qidian213/WatchNET/blob/master/images/643.png" width="800" height="600">
</div>

## Train your model

This is a training procedure using https://www.idiap.ch/dataset/unicity dataset.

### Train with unicity dataset

For each 1000 iterations, the recent weight parameters are saved as a weight file `model_iter_1000`.

```
python train.py
```

More configuration about training are in the `entity.py` file
