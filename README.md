[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Python >=3.7](https://img.shields.io/badge/Python->=3.7-yellow.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

# Corruption Invariant Learning for Re-identification

 <img src='./imgs/thumbnail_cil.png'>

The official repository for  [Benchmarks for Corruption Invariant Person Re-identification
](https://arxiv.org/abs/2111.00880) (NeurIPS 2021 Track on Datasets and Benchmarks), with exhaustive study on corruption invariant learning in single- and cross-modality ReID datasets, including [Market-1501-C](https://paperswithcode.com/dataset/market-1501-c), [CUHK03-C](https://paperswithcode.com/dataset/cuhk03-c), [MSMT17-C](https://paperswithcode.com/dataset/msmt17-c), [SYSU-MM01-C](https://paperswithcode.com/dataset/sysu-mm01-c), [RegDB-C](https://paperswithcode.com/dataset/regdb-c).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarks-for-corruption-invariant-person-re/person-re-identification-on-market-1501-c)](https://paperswithcode.com/sota/person-re-identification-on-market-1501-c?p=benchmarks-for-corruption-invariant-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarks-for-corruption-invariant-person-re/person-re-identification-on-cuhk03-c)](https://paperswithcode.com/sota/person-re-identification-on-cuhk03-c?p=benchmarks-for-corruption-invariant-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarks-for-corruption-invariant-person-re/person-re-identification-on-msmt17-c)](https://paperswithcode.com/sota/person-re-identification-on-msmt17-c?p=benchmarks-for-corruption-invariant-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarks-for-corruption-invariant-person-re/person-re-identification-on-sysu-mm01-c)](https://paperswithcode.com/sota/person-re-identification-on-sysu-mm01-c?p=benchmarks-for-corruption-invariant-person-re)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/benchmarks-for-corruption-invariant-person-re/cross-modal-person-re-identification-on-regdb-1)](https://paperswithcode.com/sota/cross-modal-person-re-identification-on-regdb-1?p=benchmarks-for-corruption-invariant-person-re)


The repo. of CIL for cross-modal ReID is [HERE](https://github.com/Zoky-2020/CIL-CrossModal)  

## Brief Introduction
When deploying person re-identification (ReID) model in safety-critical applications, it is pivotal to understanding the robustness of the model against a diverse array of image corruptions. However, current evaluations of person ReID only consider the performance on clean datasets and ignore images in various corrupted scenarios. In this work, we comprehensively establish 5 ReID benchmarks for learning corruption invariant representation. 


## Maintenance  Plan
The benchmark will be maintained by the authors. We will get constant lectures about the new proposed ReID models and evaluate them under the CIL benchmark settings in time. Besides, we gladly take feedback to the CIL benchmark and welcome any contributions in terms of the new ReID models and corresponding evaluations. Please feel free to contact us, wangzq_2021@outlook.com .

**TODO:**
- [ ] other datasets configurations
- [ ] get started tutorial
- [ ] more detailed statistical evaluations
- [ ] checkpoints of the baseline models
- [ ] cross-modality preson Re-ID dataset, CUHK-PEDES
- [ ] vehicle ReID datasets, like VehicleID, VeRi-776, etc.

(Note: codebase from [TransReID](https://github.com/heshuting555/TransReID))

## Quick Start
**1. Install dependencies**
* python=3.7.0
* pytorch=1.6.0
* torchvision=0.7.0
* timm=0.4.9
* albumentations=0.5.2
* imagecorruptions=1.1.2
* h5py=2.10.0
* cython=0.29.24
* yacs=0.1.6

**2. Prepare dataset**

Download the datasets, [Market-1501](https://openaccess.thecvf.com/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [CUHK03](https://openaccess.thecvf.com/content_cvpr_2014/html/Li_DeepReID_Deep_Filter_2014_CVPR_paper.html), [MSMT17](https://arxiv.org/abs/1711.08565). Set the root path of the dataset in `congigs/Market/resnet_base.yml`, `DATASETS: ROOT_DIR: ('root')`, or set it in `scripts/train_market.sh`, `DATASETS.ROOT_DIR "('root')"`.

**3. Train**

Train a CIL model on Market-1501,
```
sh ./scripts/train_market.sh
```

**4. Test**

Test the CIL model on Market-1501,
```
sh ./scripts/eval_market.sh
```

## Evaluating Corruption Robustness On-the-fly

#### Corruption Transform

The main code of corruption transform. (See contextual code in `./datasets/make_dataloader.py`, line 59)

```python
from imagecorruptions.corruptions import *

corruption_function = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
    glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
    elastic_transform, pixelate, jpeg_compression, speckle_noise,
    gaussian_blur, spatter, saturate, rain]
    
class corruption_transform(object):
    def __init__(self, level=0, type='all'):
        self.level = level
        self.type = type

    def __call__(self, img):
        if self.level > 0 and self.level < 6:
            level_idx = self.level
        else:
            level_idx = random.choice(range(1, 6))
        if self.type == 'all':
            corrupt_func = random.choice(corruption_function)
        else:
            func_name_list = [f.__name__ for f in corruption_function]
            corrupt_idx = func_name_list.index(self.type)
            corrupt_func = corruption_function[corrupt_idx]
        c_img = corrupt_func(img.copy(), severity=level_idx)
        img = Image.fromarray(np.uint8(c_img))
        return img
```

Evaluating corruption robustness can be realized on-the-fly by modifing the transform function uesed in test dataloader. (See details in ./datasets/make_dataloader.py, Line 266)

```python
val_with_corruption_transforms = T.Compose([
    corruption_transform(0),
    T.Resize(cfg.INPUT.SIZE_TEST),
    T.ToTensor(),])
```

#### Rain details
We introduce a rain corruption type, which is a common type of weather condition, but it is missed by the original corruption benchmark. (See details in `./datasets/make_dataloader.py`, Line 27)

```python
def rain(image, severity=1):
    if severity == 1:
        type = 'drizzle'
    elif severity == 2 or severity == 3:
        type = 'heavy'
    elif severity == 4 or severity == 5:
        type = 'torrential'
    blur_value = 2 + severity
    bright_value = -(0.05 + 0.05 * severity)
    rain = abm.Compose([
        abm.augmentations.transforms.RandomRain(rain_type=type, 
        blur_value=blur_value, brightness_coefficient=1, always_apply=True),
        abm.augmentations.transforms.RandomBrightness(limit=[bright_value, 
        bright_value], always_apply=True)])
    width, height = image.size
    if height <= 60:
        scale_factor = 65.0 / height
        new_size = (int(width * scale_factor), 65)
        image = image.resize(new_size)
    return rain(image=np.array(image))['image']
```


## Baselines
* **Single-modality datasets**
<table>
    <tr>
        <th rowspan="2"> Dataset</th>
        <th rowspan="2"> Method</th>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
    </tr>
    <tr>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
    </tr>
    <tr>
        <td rowspan="4"> Market-1501</td>
        <td> BoT </td> 
        <td> 59.30</td> <td> 85.06 </td> <td> 93.38 </td>
        <td> 0.20 </td>  <td> 8.42 </td> <td> 27.05</td>
    </tr>
    <tr>
        <td> AGW </td>
        <td> <b>64.03</b> </td> <td> 86.51 </td> <td> 94.00 </td>
        <td> 0.35</td>  <td> 12.13</td> <td> 31.90 </td>
    </tr>
    <tr>
        <td> SBS </td> 
        <td> 60.03 </td> <td> <b>88.33</b> </td> <td> <b>95.90</b> </td>
        <td> 0.29 </td>  <td> 11.54</td> <td> 34.13 </td>
    </tr>
    <tr>
        <td> <a href="https://drive.google.com/file/d/1s6TxKC0PidywnO8by34cACrzOI6MFB0R/view?usp=sharing">CIL (ours)</a> </td> 
        <td> 57.90 </td> <td> 84.04 </td> <td> 93.38 </td>
        <td> <b>1.76 (0.13)</b> </td> <td> <b>28.03 (0.45)</b> </td> <td> <b>55.57 (0.63)</b> </td>
    </tr>
    <tr>
        <td rowspan="4"> MSMT17</td>
        <td> BoT </td> 
        <td> 9.91 </td> <td> 48.34 </td> <td> 73.53 </td>
        <td> 0.07 </td>  <td> 5.28 </td> <td> 20.20 </td>
    </tr>
    <tr>
        <td> AGW </td>
        <td> 12.38 </td> <td> 51.84 </td> <td> 75.21 </td>
        <td> 0.08 </td>  <td> 6.53</td> <td> 22.77</td>
    </tr>
    <tr>
        <td> SBS </td> 
        <td> 10.26 </td> <td> <b>56.62</b> </td> <td> <b>82.02</b> </td>
        <td> 0.05</td>  <td> 7.89</td> <td> 28.77</td>
    </tr>
    <tr>
        <td> <a herf="https://drive.google.com/file/d/1JzTP0PXfme-tcsU5XTMG8FsUuZhO67QH/view?usp=sharing">CIL (ours) </a></td> 
        <td> <b>12.45</b> </td> <td> 52.40 </td> <td> 76.10 </td>
        <td> <b>0.32 (0.03)</b> </td> <td> <b>15.33 (0.20)</b> </td> <td> <b>39.79 (0.45)</b> </td>
    </tr>
    <tr>
        <td rowspan="2"> CUHK03</td>
        <td> AGW </td>
        <td> 49.97 </td> <td> 62.25 </td> <td> 64.64 </td>
        <td> 0.46 </td>  <td> 3.45</td> <td> 5.90 </td>
    </tr>
    <tr>
        <td> <a herf="https://drive.google.com/file/d/1VW6u2WB21FbFnaxV-7ZdFAcNG3IN757N/view?usp=sharing">CIL (ours) </a> </td> 
        <td> <b>53.87</b> </td> <td> <b>65.16</b> </td> <td> <b>67.29</b> </td>
        <td> <b>4.25 (0.39)</b> </td> <td> <b>16.33 (0.76)</b> </td> <td> <b>22.96 (1.04)</b> </td>
    </tr>
</table>



* **Cross-modality datasets**

**Note:** For RegDB dataset, Mode A and Mode B represent visible-to-thermal and thermal-to-visible experimental settings, respectively. And for SYSU-MM01 dataset, Mode A and Mode B represent all search and indoor search respectively. Note that we only corrupt RGB (visible) images in the corruption evaluation.

<table>
    <tr>
        <th rowspan="3"> Dataset</th>
        <th rowspan="3"> Method</th>
        <th colspan="6">Mode A</th>
        <th colspan="6">Mode B</th>
    </tr>
    <tr>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
    </tr>
    <tr>
        <th>mINP</th> <th>mAP</th> <th>R-1</th>
        <th>mINP</th> <th>mAP</th> <th>R-1</th>
        <th>mINP</th> <th>mAP</th> <th>R-1</th>
        <th>mINP</th> <th>mAP</th> <th>R-1</th>
    </tr>
    <tr>
        <td rowspan="2"> SYSU-MM01</td>
        <td> AGW </td>
        <td> 36.17 </td> <td> <b>47.65</b> </td> <td> <b>47.50</b> </td>
        <td> 14.73 </td>  <td> 29.99 </td> <td> 34.42 </td>
        <td> <b>59.74</b> </td> <td> <b>62.97</b> </td> <td> <b>54.17</b> </td>
        <td> 35.39 </td>  <td> 40.98 </td> <td> 33.80 </td>
    </tr>
    <tr>
        <td> <a herf="https://drive.google.com/file/d/1GeK7xeE6L_6ZbypvhUdSyqcZwxP4or7O/view?usp=sharing">CIL (ours) </a></td> 
        <td> <b>38.15</b> </td> <td> 47.64 </td> <td> 45.51 </td>
        <td> <b>22.48 (1.65)</b> </td>  <td> <b>35.92 (1.22)</b> </td> <td> <b>36.95 (0.67)</b> </td>
        <td> 57.41 </td> <td> 60.45 </td> <td> 50.98 </td>
        <td> <b>43.11 (4.19)</b> </td>  <td> <b>48.65 (4.57)</b> </td> <td> <b>40.73 (5.55)</b> </td>
    </tr>
    <tr>
        <td rowspan="2"> RegDB</td>
        <td> AGW </td>
        <td> 54.10 </td> <td> 68.82 </td> <td> <b>75.78</b> </td>
        <td> 32.88 </td>  <td> 43.09 </td> <td> 45.44 </td>
        <td> 52.40 </td> <td> 68.15 </td> <td> <b>75.29</b> </td>
        <td> 6.00 </td>  <td> 41.37 </td> <td> <b>67.54</b> </td>
    </tr>
    <tr>
        <td> <a herf="https://drive.google.com/file/d/1xn5Kk3sIvXie-2FJGXX4Lwjhj56bJRsl/view?usp=sharing">CIL (ours) </a> </td> 
        <td> <b>55.68</b> </td> <td> <b>69.75</b> </td> <td> 74.96 </td>
        <td> <b>38.66 (0.01)</b> </td>  <td> <b>49.76 (0.03)</b> </td> <td> <b>52.25 (0.03)</b> </td>
        <td> <b>55.50</b> </td> <td> <b>69.21</b> </td> <td> 74.95 </td>
        <td> <b>11.94 (0.12)</b> </td>  <td> <b>47.90 (0.01)</b> </td> <td> 67.17 (0.06)</td>
    </tr>
</table>

(Note: the checkpoints are provided [here](https://drive.google.com/drive/folders/1eSP6LMUh12oHejvzLD2qH0iZ-jTxVPBk?usp=sharing).)
## Recent Advance in Person Re-ID

<table><tr>
<td> <img src='./imgs/market.png' width=100%> </td>
<td> <img src='./imgs/market_c.png' width=100%> </td>
</tr></table>

## Leaderboard

#### Market1501-C 
**(Note: ranked by mAP on corrupted test set)**
<table>
    <tr>
        <th rowspan="2"> Method</th>
        <th rowspan="2"> Reference </th>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
    </tr>
    <tr>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
    </tr>
    <tr>
        <td> TransReID </td> 
        <td> <a href="https://arxiv.org/abs/2102.04378"> Shuting He et al. (2021) </a> </td>
        <td> 69.29 </td> <td> 88.93 </td> <td> 95.07 </td>
        <td> 1.98 </td>  <td> <b>27.38</b> </td> <td> 53.19 </td>
    </tr>
    <tr>
        <td> CaceNet </td>
        <td> <a href="https://arxiv.org/abs/2009.05250"> Fufu Yu et al. (2020) </a> </td>
        <td> 70.47 </td> <td> 89.82 </td> <td> 95.40 </td>
        <td> 0.67 </td>  <td> <b>18.24</b> </td> <td> 42.92 </td>
    </tr>
    <tr>
        <td> LightMBN </td> 
        <td> <a href="https://arxiv.org/abs/2101.10774"> Fabian Herzog et al. (2021) </a> </td>
        <td> 73.29 </td> <td> 91.54 </td> <td> 96.53 </td>
        <td> 0.50 </td> <td> <b>14.84</b> </td> <td> 38.68 </td>
    </tr>
    <tr>
        <td> PLR-OS </td> 
        <td> <a href="https://arxiv.org/abs/2001.07442"> Ben Xie et al. (2020) </a> </td>
        <td> 66.42 </td> <td> 88.93 </td> <td> 95.19 </td>
        <td> 0.48 </td> <td> <b>14.23</b> </td> <td> 37.56 </td>
    </tr>
    <tr>
        <td> RRID </td> 
        <td> <a href="https://arxiv.org/abs/1911.09318"> Hyunjong Park et al. (2019) </a> </td>
        <td> 67.14 </td> <td> 88.43 </td> <td> 95.19 </td>
        <td> 0.46 </td> <td> <b>13.45</b> </td> <td> 36.57 </td>
    </tr>
    <tr>
        <td> Pyramid </td> 
        <td> <a href="https://arxiv.org/abs/1810.12193"> Feng Zheng et al. (2018) </a> </td>
        <td> 61.61 </td> <td> 87.50 </td> <td> 94.86 </td>
        <td> 0.36 </td> <td> <b>12.75</b> </td> <td> 35.72 </td>
    </tr>
    <tr>
        <td> PCB </td> 
        <td> <a href="https://arxiv.org/abs/1711.09349"> Yifan Sun et al.(2017) </a> </td>
        <td> 41.97 </td> <td> 82.19 </td> <td> 94.15 </td>
        <td> 0.41 </td>  <td> <b>12.72</b> </td> <td> 34.93 </td>
    </tr>
    <tr>
        <td> BDB </td> 
        <td> <a href="https://arxiv.org/abs/1811.07130"> Zuozhuo Dai et al. (2018) </a> </td>
        <td> 61.78 </td> <td> 85.47 </td> <td> 94.63 </td>
        <td> 0.32 </td> <td> <b>10.95</b> </td> <td> 33.79 </td>
    </tr>
    <tr>
        <td> Aligned++ </td> 
        <td> <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320319302031"> Hao Luo et al. (2019) </a> </td>
        <td> 47.31 </td> <td> 79.10 </td> <td> 91.83 </td>
        <td> 0.32 </td> <td> <b>10.95</b> </td> <td> 31.00 </td>
    </tr>
    <tr>
        <td> AGW </td> 
        <td> <a href="https://arxiv.org/abs/2001.04193v2"> Mang Ye et al. (2020) </a> </td>
        <td> 65.40 </td> <td> 88.10 </td> <td> 95.00 </td>
        <td> 0.30 </td> <td> <b>10.80</b> </td> <td> 33.40 </td>
    </tr>
    <tr>
        <td> MHN </td> 
        <td> <a href="https://arxiv.org/abs/1908.05819"> Binghui Chen et al. (2019) </a> </td>
        <td> 55.27 </td> <td> 85.33 </td> <td> 94.50 </td>
        <td> 0.38 </td> <td> <b>10.69</b> </td> <td> 33.29 </td>
    </tr>
    <tr>
        <td> LUPerson </td> 
        <td> <a href="https://arxiv.org/abs/2012.03753"> Dengpan Fu et al. (2020) </a> </td>
        <td> 68.71 </td> <td> 90.32 </td> <td> 96.32 </td>
        <td> 0.29 </td> <td> <b>10.37</b> </td> <td> 32.22 </td>
    </tr>
    <tr>
        <td> OS-Net </td> 
        <td> <a href="https://arxiv.org/abs/1905.00953"> Kaiyang Zhou et al. (2019) </a> </td>
        <td> 56.78 </td> <td> 85.67 </td> <td> 94.69 </td>
        <td> 0.23 </td> <td> <b>10.37</b> </td> <td> 30.94 </td>
    </tr>
    <tr>
        <td> VPM </td> 
        <td> <a href="https://arxiv.org/abs/1904.00537"> Yifan Sun et al. (2019) </a> </td>
        <td> 50.09 </td> <td> 81.43 </td> <td> 93.79 </td>
        <td> 0.31 </td> <td> <b>10.15</b> </td> <td> 31.17 </td>
    </tr>
    <tr>
        <td> DG-Net </td> 
        <td> <a href="https://arxiv.org/abs/1904.07223"> Zhedong Zheng et al. (2019) </a> </td>
        <td> 61.60 </td> <td> 86.09 </td> <td> 94.77 </td>
        <td> 0.35 </td> <td> <b>9.96</b> </td> <td> 31.75 </td>
    </tr>
    <tr>
        <td> ABD-Net </td> 
        <td> <a href="https://arxiv.org/abs/1908.01114"> Tianlong Chen et al. (2019) </a> </td>
        <td> 64.72 </td> <td> 87.94 </td> <td> 94.98 </td>
        <td> 0.26 </td> <td> <b>9.81</b> </td> <td> 29.65 </td>
    </tr>
    <tr>
        <td> MGN </td> 
        <td> <a href="https://arxiv.org/abs/1804.01438"> Guanshuo Wang et al.(2018) </a> </td>
        <td> 60.86 </td> <td> 86.51 </td> <td> 93.88 </td>
        <td> 0.29 </td> <td> <b>9.72</b> </td> <td> 29.56 </td>
    </tr>
    <tr>
        <td> F-LGPR </td> 
        <td> <a href="https://arxiv.org/abs/2101.08783"> Yunpeng Gong et al. (2021) </a> </td>
        <td> 65.48 </td> <td> 88.22 </td> <td> 95.37 </td>
        <td> 0.23 </td> <td> <b>9.08</b> </td> <td> 29.35 </td>
    </tr>
    <tr>
        <td> TDB </td> 
        <td> <a href="https://arxiv.org/abs/2010.05435"> Rodolfo Quispe et al. (2020) </a> </td>
        <td> 56.41 </td> <td> 85.77 </td> <td> 94.30 </td>
        <td> 0.20 </td> <td> <b>8.90</b> </td> <td> 28.56 </td>
    </tr>
    <tr>
        <td> LGPR </td> 
        <td> <a href="https://arxiv.org/abs/2101.08783"> Yunpeng Gong et al. (2021) </a> </td>
        <td> 58.71 </td> <td> 86.09 </td> <td> 94.51 </td>
        <td> 0.24 </td> <td> <b>8.26</b> </td> <td> 27.72 </td>
    </tr>
    <tr>
        <td> BoT </td> 
        <td> <a href="https://arxiv.org/abs/1906.08332"> Hao Luo et al. (2019) </a> </td>
        <td> 51.00 </td> <td> 83.90 </td> <td> 94.30 </td>
        <td> 0.10 </td> <td> <b>6.60</b> </td> <td> 26.20 </td>
    </tr>
</table>


#### CUHK03-C (detected)
**(Note: ranked by mAP on corrupted test set)**
<table>
    <tr>
        <th rowspan="2"> Method</th>
        <th rowspan="2"> Reference </th>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
    </tr>
    <tr>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
    </tr>
    <tr>
        <td> CaceNet </td> 
        <td> <a href="https://arxiv.org/abs/2009.05250"> Fufu Yu et al. (2020) </a> </td>
        <td> 65.22 </td> <td> 75.13 </td> <td> 77.64 </td>
        <td> 2.09 </td>  <td> <b>10.62</b> </td> <td> 17.04 </td>
    </tr>
    <tr>
        <td> Pyramid </td>
        <td> <a href="https://arxiv.org/abs/1810.12193"> Feng Zheng et al. (2018) </a> </td>
        <td> 61.41 </td> <td> 73.14 </td> <td> 79.54 </td>
        <td> 1.10 </td>  <td> <b>8.03</b> </td> <td> 10.42 </td>
    </tr>
    <tr>
        <td> RRID </td> 
        <td> <a href="https://arxiv.org/abs/1911.09318"> Hyunjong Park et al. (2019) </a> </td>
        <td> 55.81 </td> <td> 67.63 </td> <td> 74.99 </td>
        <td> 1.00 </td> <td> <b>7.30</b> </td> <td> 9.66 </td>
    </tr>
    <tr>
        <td> PLR-OS </td> 
        <td> <a href="https://arxiv.org/abs/2001.07442"> Ben Xie et al. (2020) </a> </td>
        <td> 62.72 </td> <td> 74.67 </td> <td> 78.14 </td>
        <td> 0.89 </td> <td> <b>6.49</b> </td> <td> 10.99 </td>
    </tr>
    <tr>
        <td> Aligned++ </td> 
        <td> <a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320319302031"> Hao Luo et al. (2019) </a> </td>
        <td> 47.32 </td> <td> 59.76 </td> <td> 62.07 </td>
        <td> 0.56 </td> <td> <b>4.87</b> </td> <td> 7.99 </td>
    </tr>
    <tr>
        <td> MGN </td> 
        <td> <a href="https://arxiv.org/abs/1804.01438"> Guanshuo Wang et al.(2018) </a> </td>
        <td> 51.18 </td> <td> 62.73 </td> <td> 69.14 </td>
        <td> 0.46 </td> <td> <b>4.20</b> </td> <td> 5.44 </td>
    </tr>
    <tr>
        <td> MHN </td> 
        <td> <a href="https://arxiv.org/abs/1908.05819"> Binghui Chen et al. (2019) </a> </td>
        <td> 56.52 </td> <td> 66.77 </td> <td> 72.21 </td>
        <td> 0.46 </td> <td> <b>3.97</b> </td> <td> 8.27 </td>
    </tr>
</table>


#### MSMT17-C (Version 2)
**(Note: ranked by mAP on corrupted test set)**
<table>
    <tr>
        <th rowspan="2"> Method</th>
        <th rowspan="2"> Reference </th>
        <th colspan="3">Clean Eval.</th>
        <th colspan="3">Corruption Eval.</th>
    </tr>
    <tr>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
        <th>mINP</th> <th>mAP</th> <th>Rank-1</th>
    </tr>
    <tr>
        <td> OS-Net </td> 
        <td> <a href="https://arxiv.org/abs/1905.00953"> Kaiyang Zhou et al. (2019) </a> </td>
        <td> 4.05 </td> <td> 40.05 </td> <td> 71.86 </td>
        <td> 0.08 </td> <td> <b>7.86</b> </td> <td> 28.51 </td>
    </tr>
    <tr>
        <td> AGW </td> 
        <td> <a href="https://arxiv.org/abs/2001.04193v2"> Mang Ye et al. (2020) </a> </td>
        <td> 12.38 </td> <td> 51.84 </td> <td> 75.21 </td>
        <td> 0.08 </td> <td> <b>6.53</b> </td> <td> 22.77 </td>
    </tr>
    <tr>
        <td> BoT </td> 
        <td> <a href="https://arxiv.org/abs/1906.08332"> Hao Luo et al. (2019) </a> </td>
        <td> 9.91 </td> <td> 48.34 </td> <td> 73.53 </td>
        <td> 0.07 </td> <td> <b>5.28</b> </td> <td> 20.20 </td>
    </tr>
</table>

## Citation

Kindly include a reference to this paper in your publications if it helps your research:
```
@misc{chen2021benchmarks,
    title={Benchmarks for Corruption Invariant Person Re-identification},
    author={Minghui Chen and Zhiqiang Wang and Feng Zheng},
    year={2021},
    eprint={2111.00880},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

