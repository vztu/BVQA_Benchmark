# BVQA_Benchmark

This is a resource list for blind video quality assessment (BVQA) models on user-generated databases, i.e., the UGC-VQA problem studied in our paper [IEEE TIP2021] [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354). [IEEEXplore](https://ieeexplore.ieee.org/document/9405420)

The following content include datasets, models & codes, performance benchmark and leaderboard. 

**Maintained by:**  [Zhengzhong Tu](mailto:zhengzhong.tu@utexas.edu)

:+1: **Any suggestion or idea is welcomed.** Please see [Contributing](#contributing)

- Updates [10-21-2021] All the features I used in the paper can be downloaded here: [Google Drive](https://drive.google.com/drive/folders/1_HFMO1KflNvlwkLC02ZWkxxiKZtFQqwV?usp=sharing) 


## Contents

- [BVQA_Benchmark](#bvqa_benchmark)
  - [Contents](#contents)
  - [Evaluate Your Own Model](#evaluate-your-own-model)
      - [Pre-requisites](#pre-requisites)
      - [Demo evaluations (BRISUQE on KoNViD-1k)](#demo-evaluations-brisuqe-on-konvid-1k)
      - [Custom usage with your own model on given dataset](#custom-usage-with-your-own-model-on-given-dataset)
  - [UGC-VQA Datasets](#ugc-vqa-datasets)
  - [BIQA / BVQA Models <a name="biqa-bvqa-model"></a>](#biqa--bvqa-models-)
      - [BIQA](#biqa)
      - [BVQA](#bvqa)
  - [Performance Benchmark](#performance-benchmark)
    - [Regression Results](#regression-results)
  - [Contributing](#contributing)
  - [Citation](#citation)


## Evaluate Your Own Model

Extract features in the form of NxM matrix (N:#samples, M:#features) on a given VQA dataset and save it in `data/` directory. Let metadata file be stored in the same folder with MOSs in the same order as your feature matrix (We have already provided the MOS arrays of three UGC datasets). The `evaluate_bvqa_features.py` evaluates the extracted features via 100 random train-test splits and reports the median (std) SRCC/KRCC/PLCC/RMSE performances. Note that it is not applicable to deep learning models (feature-based model only).

#### Pre-requisites

- python3
- sklearn

#### Demo evaluations (BRISUQE on KoNViD-1k)
```
$ python3 src/evaluate_bvqa_features.py
```

#### Custom usage with your own model on given dataset
```
$ python3 src/evaluate_bvqa_features.py [-h] [--model_name MODEL_NAME]
                                   [--dataset_name DATASET_NAME]
                                   [--feature_file FEATURE_FILE]
                                   [--mos_file MOS_FILE] [--out_file OUT_FILE]
                                   [--color_only] [--log_short] [--use_parallel]
                                   [--num_iterations NUM_ITERATIONS]
                                   [--max_thread_count MAX_THREAD_COUNT]
```

## UGC-VQA Datasets

| BVQA Dataset | Download | Paper |
|:----:|:----:|:----:|
| **KoNViD-1k (2017)** |  [KoNViD-1k](http://database.mmsp-kn.de/konvid-1k-database.html)  | [Hosu et al. QoMEX'17](https://datasets.vqa.mmsp-kn.de/archives/papers/Hosu-Konvid-1k.pdf)
| **LIVE-VQC (2018)** |  [LIVE-VQC](https://live.ece.utexas.edu/research/LIVEVQC/index.html)  | [Sinno et al. TIP'19](https://arxiv.org/pdf/1803.01761.pdf)
| **YouTube-UGC (2019)** | [YouTube-UGC](https://media.withyoutube.com/) | [Wang et al. MMSP'19](https://arxiv.org/abs/1904.06457)
| **LIVE-FB-LSVQ (2021)** | [LIVE-FB-LSVQ](https://github.com/baidut/PatchVQ) | [Ying et al. CVPR'21](https://arxiv.org/abs/2011.13544)


## BIQA / BVQA Models <a name="biqa-bvqa-model"></a>

#### BIQA

|    Model   | Download            | Paper             |
|:------------:|:-------------------:|:-------------------:|
|  BRISQUE    | [BRISQUE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'12](https://www.live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) |
|  NIQE       | [NIQE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'13](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.299.1429&rep=rep1&type=pdf)
| ILNIQE      | [ILNIQE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Zhang et al. TIP'15](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.342&rep=rep1&type=pdf)
| GM-LOG      | [GM-LOG](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Xue et al. TIP'14](https://live.ece.utexas.edu/publications/2014/BIQAUsingGM-LoG.pdf)
| HIGRADE     | [HIGRADE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Kundu et al. TIP'17](https://ieeexplore.ieee.org/abstract/document/7885070)
| FRIQUEE     | [FRIQUEE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Ghadiyaram et al. JoV'17](https://jov.arvojournals.org/article.aspx?articleid=2599945)
| CORNIA      | [BIQA_Toolbox](https://github.com/HuiZeng/BIQA_Toolbox) | [Ye et al. CVPR'12](https://ieeexplore.ieee.org/abstract/document/6247789)
| HOSA        | [BIQA_Toolbox](https://github.com/HuiZeng/BIQA_Toolbox) | [Xu et al. TIP'16](https://ieeexplore.ieee.org/abstract/document/7501619)
| KonCept 512 | [koniq](https://github.com/subpic/koniq), [koniq-PyTorch](https://github.com/ZhengyuZhao/koniq-PyTorch) | [Hosu et al. TIP'20](https://arxiv.org/abs/1910.06180) |
| PaQ-2-PiQ   | [PaQ-2-PiQ](https://github.com/baidut/PaQ-2-PiQ), [paq2piq-PyTorch](https://github.com/baidut/paq2piq) | [Ying et al. CVPR'20](https://arxiv.org/abs/1912.10088) |

#### BVQA

|    Model   | Download            | Paper             |
|:------------:|:-------------------:|:-------------------:|
| VIIDEO      | [VIIDEO](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'16](https://utw10503.utweb.utexas.edu/publications/2016/07332944.pdf)
| V-BLIINDS   | [V-BLIINDS](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Saad et al. TIP'14](https://www.live.ece.utexas.edu/publications/2014/VideoBLIINDS.pdf)
| TLVQM       | [nr-vqa-consumervideo](https://github.com/jarikorhonen/nr-vqa-consumervideo) | [Korhenen et al. TIP'19](https://ieeexplore.ieee.org/document/8742797)
| VSFA        | [VSFA](https://github.com/lidq92/VSFA)  | [Li et al. MM'19](https://dl.acm.org/doi/abs/10.1145/3343031.3351028)
| NSTSS       | [NRVQA-NSTSS](https://github.com/lfovia/NRVQA-NSTSS) | [Dendi et al. TIP'20](https://ieeexplore.ieee.org/abstract/document/9059006)
| VIDEVAL     | [VIDEVAL](https://github.com/vztu/VIDEVAL_release) | [Tu et al. TIP'21](https://ieeexplore.ieee.org/document/9405420)
| MDTVSFA     | [MDTVSFA](https://github.com/lidq92/MDTVSFA) | [Li et al. IJCV'21](https://link.springer.com/article/10.1007/s11263-020-01408-w)
| RAPIQUE      | [RAPIQUE](https://github.com/vztu/RAPIQUE) | [Tu et al. OJSP'21](https://ieeexplore.ieee.org/document/9463703/)
| PatchVQ      | [PatchVQ](https://github.com/baidut/PatchVQ) | [Ying et al. CVPR'21](https://arxiv.org/abs/2011.13544) 
| CoINVQ       | [CoINVQ]()  | [Wang et al. CVPR'21](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Rich_Features_for_Perceptual_Quality_Assessment_of_UGC_Videos_CVPR_2021_paper.html)

## Performance Benchmark

### Regression Results

Median SRCC (std SRCC) of 100 random train-test (80%-20%) splits.

|    Methods   | KoNViD-1k             | LIVE-VQC             | YouTube-UGC         | All-Combined |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.6567 (0.0351)  | 0.5925 (0.0681)     | 0.3820 (0.0519) | 0.5695 (0.0289) |
| NIQE         | 0.5417 (0.0347)  | 0.5957 (0.0571)     | 0.2379 (0.0487) | 0.4622 (0.0313) |
| IL-NIQE      | 0.5264 (0.0294)  | 0.5037 (0.0712)     | 0.2918 (0.0502) | 0.4592 (0.0307) |
| GM-LOG       | 0.6578 (0.0324)  | 0.5881 (0.0683)     | 0.3678 (0.0589) | 0.5650 (0.0295) |
| HIGRADE      | 0.7206 (0.0302)  | 0.6103 (0.0680)     | 0.7376 (0.0338) | 0.7398 (0.0189) |
| FRIQUEE      | 0.7472 (0.0263)  | 0.6579 (0.0536)     | 0.7652 (0.0301) | 0.7568 (0.0237) |
| CORNIA       | 0.7169 (0.0245)  | 0.6719 (0.0473)     | 0.5972 (0.0413) | 0.6764 (0.0216) |
| HOSA         | 0.7654 (0.0224)  | 0.6873 (0.0462)     | 0.6025 (0.0344) | 0.6957 (0.0180) |
| VGG-19       | 0.7741 (0.0288)  | 0.6568 (0.0536) | 0.7025 (0.0281) | 0.7321 (0.0180) |
| ResNet-50    | 0.8018 (0.0255)  | 0.6636 (0.0511) | 0.7183 (0.0281) | 0.7557 (0.0177) |
| KonCept512   | 0.7349 (0.0252)  | 0.6645 (0.0523)     | 0.5872 (0.0396) | 0.6608 (0.0221) |
| PaQ-2-PiQ    | 0.6130 (0.0325)  | 0.6436 (0.0457)     | 0.2658 (0.0473) | 0.4727 (0.0298) |
| VIIDEO       | 0.2988 (0.0561)  | 0.0332 (0.0856)     | 0.0580 (0.0536) | 0.1039 (0.0349) |
| V-BLIINDS    | 0.7101 (0.0314)  | 0.6939 (0.0502)    | 0.5590 (0.0496) | 0.6545 (0.0232) |
| TLVQM        | 0.7729 (0.0242)  | 0.7988 (0.0365)    | 0.6693 (0.0306) | 0.7271 (0.0189) | 
| VIDEVAL      | 0.7832 (0.0216) | 0.7522 (0.0390)  | 0.7787 (0.0254) | 0.7960 (0.0151) |
| VSFA | 0.755 (0.025) | - | - | - |
| NSTSS | 0.6417 | - | - | - |
| VIDEVAL+KonCept512 | 0.8149 (0.0194) | 0.7849 (0.0440) | 0.8083 (0.0232) | 0.8123 (0.0163) |
| MDTVSFA  | 0.7812 (0.0278)   | 0.7382 (0.0357)  |  -  |  - | 
| PatchVQ  | 0.791  | 0.827  | - |  - |
| CoINVQ   | 0.767  | -  | 0.816  | - |

<!-- | VIDEVAL+PaQ-2-PiQ | 0.7844 (0.0213) | 0.7677 (0.0403) | 0.7981 (0.0212) | 0.7962 (0.0163) | -->
<!-- | VIDEVAL+VGG-19 | 0.7827 (0.0296)   | 0.7274 (0.0489) | 0.7868 (0.0216) | 0.7859 (0.0161) | 
| VIDEVAL+ResNet-50 | 0.8129 (0.0285) | 0.7456 (0.0454) | 0.8085 (0.0205) | 0.8115 (0.8286) | -->


The median PLCC (std PLCC) of 100 random train-test (80%-20%) splits.

|    Model   | KoNViD-1k             | LIVE-VQC             | YouTube-UGC         | All-Combined |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:------------:|
| BRISQUE      | 0.6576 (0.0342)  | 0.6380 (0.0632)     | 0.3952 (0.0486) | 0.5861 (0.0272) |
| NIQE         | 0.5530 (0.0337)  | 0.6286 (0.0512)     | 0.2776 (0.0431) | 0.4773 (0.0287) |
| IL-NIQE      | 0.5400 (0.0337)  | 0.5437 (0.0707)     | 0.3302 (0.0579) | 0.4741 (0.0280) |
| GM-LOG       | 0.6636 (0.0315)  | 0.6212 (0.0636)     | 0.3920 (0.0549) | 0.5942 (0.0306) |
| HIGRADE      | 0.7269 (0.0287)  | 0.6332 (0.0652)     | 0.7216 (0.0334) | 0.7368 (0.0190) |
| FRIQUEE      | 0.7482 (0.0257)  | 0.7000 (0.0587)     | 0.7571 (0.0324) | 0.7550 (0.0226) |
| CORNIA       | 0.7135 (0.0236)  | 0.7183 (0.0420)     | 0.6057 (0.0399) | 0.6974 (0.0202) |
| HOSA         | 0.7664 (0.0207)  | 0.7414 (0.0410)     | 0.6047 (0.0347) | 0.7082 (0.0167) |
| VGG-19       | 0.7845 (0.0246)  | 0.7160 (0.0481)     | 0.6997 (0.0281) | 0.7482 (0.0176) |
| ResNet-50    | 0.8104 (0.0229)  | 0.7205 (0.0434)     | 0.7097 (0.0276) | 0.7747 (0.0167) |
| KonCept512   | 0.7489 (0.0240)  | 0.7278 (0.0464)     | 0.5940 (0.0412) | 0.6763 (0.0227) |
| PaQ-2-PiQ    | 0.6014 (0.0338)  | 0.6683 (0.0445)     | 0.2935 (0.0490) | 0.4828 (0.0293) |
| VIIDEO       | 0.3002 (0.0539)  | 0.2146 (0.0903)     | 0.1534 (0.0498) | 0.1621 (0.0355) |
| V-BLIINDS    | 0.7037 (0.0301)  | 0.7178 (0.0500)     | 0.5551 (0.0465) | 0.6599 (0.0234) |
| TLVQM        | 0.7688 (0.0238)  | 0.8025 (0.0360)     | 0.6590 (0.0302) | 0.7342 (0.0180) | 
| VIDEVAL      | 0.7803 (0.0223)  | 0.7514 (0.0420)     | 0.7733 (0.0257) | 0.7939 (0.0157)} |
| VSFA | 0.744 (0.029) | - | - | - |
| NSTSS | 0.6531 | - | - | - |
| VIDEVAL+KonCept512 | 0.8169 (0.0179)  | 0.8010 (0.0398) | 0.8028 (0.0234) | 0.8168 (0.0128) |
| MDTVSFA  |  0.7856 (0.0240) | 0.7728 (0.0351)  |  -  |  -  |
| PatchVQ  | 0.786  | 0.837 | - | - |
| CoINVQ   | 0.764  | -  | 0.802  | - |

<!-- | VIDEVAL+PaQ-2-PiQ | 0.7793 (0.0226) | 0.7686 (0.0411) | 0.7941 (0.0224) | 0.7934 (0.0157) | -->
<!-- | VIDEVAL+VGG-19    | 0.7913 (0.0253)  | 0.7717 (0.0431)  | 0.7847 (0.0212)  | 0.7962 (0.0142) | 
| VIDEVAL+ResNet-50 | 0.8200 (0.0238)  | 0.7810 (0.0434) | 0.8033 (0.0208)  | 0.8286 (0.0128) |  -->


## Contributing
Please feel free to send an [issue](https://github.com/vztu/BVQA_Benchmark/issues) or [pull requests](https://github.com/vztu/BVQA_Benchmark/pulls) or email [me](mailto:zhengzhong.tu@utexas.edu) to add links or new results.

## Citation

Should you find this repo useful to your research, we sincerely appreciate it if you cite our papers:blush::

```
@article{tu2020ugc,
  title={UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content},
  author={Tu, Zhengzhong and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={arXiv preprint arXiv:2005.14354},
  year={2020}
}

@inproceedings{tu2020comparative,
  title={A Comparative Evaluation Of Temporal Pooling Methods For Blind Video Quality Assessment}, 
  author={Z. {Tu} and C. -J. {Chen} and L. -H. {Chen} and N. {Birkbeck} and B. {Adsumilli} and A. C. {Bovik}},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},  
  year={2020},
  pages={141-145},
  doi={10.1109/ICIP40778.2020.9191169}
}
```
