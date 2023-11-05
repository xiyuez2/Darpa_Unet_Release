# Unet model for Darpa poly segmentation project
- Getting start for the Semantic Segmentation Competition quickly

## Requirements
- Python >= 3.7 (3.8 recommended)
- Pytorch >= 1.7 (conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch)
- pytorch_lightning
- tqdm
- albumentations
- opencv-python
- others -- please see env.yaml

## Usage
test.py and inference.py can run a pretrained model on data here is the usage for inference.py:

```
mkdir [folder to save results] 
python inference.py --mapPath [path to a h5 file for patched map and legend] --outputPath [folder to save results] --modelPath [path to the model ckpt file]
```

for example:
```
mkdir res
python inference.py --mapPath "/projects/bbym/shared/data/commonPatchData/256/OK_250K.hdf5" --outputPath res --modelPath jaccard_index_value\=0.9229.ckpt
```
