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
Please enter the "ckpts" folder and then execute the following unzip command to get the pre-trained model "MSAN.pth".
```
cat MSAN.zip.* > MSAN.zip
unzip MSAN.zip
```

Please copy test samples into './test_samples'. Then running the following command.
```
python test.py
```

## Citation
If you think this work is useful for your research, please cite the following paper.

```
@article{guo2022multi,
  title={Multi-Stage Attentive Network for Motion Deblurring via Binary Cross-Entropy Loss},
  author={Guo, Cai and Chen, Xinan and Chen, Yanhua and Yu, Chuying},
  journal={Entropy},
  volume={24},
  number={10},
  pages={1414},
  year={2022},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
