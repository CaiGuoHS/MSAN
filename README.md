# Multi-Stage Attentive Network for Motion Deblurring via Binary Cross-Entropy Loss
by Cai Guo, Xinnan Chen, Yanhua Chen, Chuying Yu.

## Dependencies
python
```
conda create -n msan python=3.8
conda activate msan
```
pytorch
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

## Testing
Please enter the ckpts folder and execute the following unzip command.
```
cat MSAN.zip.* > MSAN.zip
unzip MSAN.zip
```

Please copy test samples into './test_samples'. Then running the following command.
```
python test.py
```
