# BVQA_Benchmark
A performance benchmark for blind video quality assessment (BVQA) models on user-generated databases, for the UGC-VQA problem studied in the paper [UGC-VQA: Benchmarking blind video quality assessment for user generated content](https://arxiv.org/abs/2005.14354).

## Pre-requisites

- \>= python 3.6.7
- sklearn

## Usage (feature-based model only)

Compute feature matrix on a given dataset and put it in `data/` folder, with MOS array stored in the same order (We have provided the MOS arrays of three UGC datasets). The code evaluates the extracted features through 100 iterations of train-test splits and returns the median (std) SRCC/KRCC/PLCC/RMSE performances. Note that it is not applicable for deep learning based models.

##### Run demo (BRISUQE x KoNViD-1k)
```
$ python src/evaluate_bvqa_features.py
```

##### Custom usage
```
$ python src/evaluate_bvqa_features.py [-h] [--model_name MODEL_NAME]
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


## Evaluated BIQA/BVQA Models

|    Methods   | Download            | Paper             |
|:------------:|:-------------------:|:-------------------:|
|  BRISQUE    | [BRISQUE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'12](https://www.live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf) |
|  NIQE       | [NIQE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'13](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.299.1429&rep=rep1&type=pdf)
| ILNIQE      | [ILNIQE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Zhang et al. TIP'15](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.342&rep=rep1&type=pdf)
| GM-LOG      | [GM-LOG](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Xue et al. TIP'14](https://live.ece.utexas.edu/publications/2014/BIQAUsingGM-LoG.pdf)
| HIGRADE     | [HIGRADE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Kundu et al. TIP'17](https://ieeexplore.ieee.org/abstract/document/7885070)
| FRIQUEE     | [FRIQUEE](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Ghadiyaram et al. JoV'17](https://jov.arvojournals.org/article.aspx?articleid=2599945)
| CORNIA      | [BIQC_Toolbox](https://github.com/HuiZeng/BIQA_Toolbox) | [Ye et al. CVPR'12](https://ieeexplore.ieee.org/abstract/document/6247789)
| HOSA        | [BIQA_Toolbox](https://github.com/HuiZeng/BIQA_Toolbox) | [Xu et al. TIP'16](https://ieeexplore.ieee.org/abstract/document/7501619)
| VIIDEO      | [VIIDEO](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Mittal et al. TIP'16](https://utw10503.utweb.utexas.edu/publications/2016/07332944.pdf)
| V-BLIINDS   | [V-BLIINDS](https://live.ece.utexas.edu/research/Quality/index_algorithms.htm) | [Saad et al. TIP'14](https://www.live.ece.utexas.edu/publications/2014/VideoBLIINDS.pdf)
| TLVQM       | [nr-vqa-consumervideo](https://github.com/jarikorhonen/nr-vqa-consumervideo) | [Korhenen et al. TIP'19](https://ieeexplore.ieee.org/document/8742797)
| VIDEVAL     | [VIDEVAL_release](https://github.com/tu184044109/VIDEVAL_release) | [Tu et al. CoRR'20](https://arxiv.org/abs/2005.14354)

## Results
The median SRCC (std SRCC) of 100 repititions of different methods on different datasets.

|    Methods   | KoNViD-1k             | LIVE-VQC             | YouTube-UGC         |
|:------------:|:---------------------:|:--------------------:|:-------------------:|
| BRISQUE      | 0.6567 (0.0351)  | 0.5925 (0.0681)     | 0.3820 (0.0519) |
| NIQE         | 0.5417 (0.0347)  | 0.5957 (0.0571)     | 0.2379 (0.0487) |
| IL-NIQE      | 0.5264 (0.0294)  | 0.5037 (0.0712)     | 0.2918 (0.0502) |
| GM-LOG       | 0.6578 (0.0324)  | 0.5881 (0.0683)     | 0.3678 (0.0589) |
| HIGRADE      | 0.7206 (0.0302)  | 0.6103 (0.0680)     | **0.7376 (0.0338)** |
| FRIQUEE      | 0.7472 (0.0263)  | 0.6579 (0.0536)     | **0.7652 (0.0301)** |
| CORNIA       | 0.7169 (0.0245)  | 0.6719 (0.0473)     | 0.5972 (0.0413) |
| HOSA         | 0.7654 (0.0224)  | 0.6873 (0.0462)     | 0.6025 (0.0344) |
| VGG-19       | **0.7741 (0.0288)**  | 0.6568 (0.0536) | 0.7025 (0.0281) |
| ResNet-50    | **0.8018 (0.0255)**  | 0.6636 (0.0511) | 0.7183 (0.0281) |
| VIIDEO       | 0.2988 (0.0561)  | 0.0332 (0.0856)     | 0.0580 (0.0536) |
| V-BLIINDS    | 0.7101 (0.0314)  | **0.6939 (0.0502)**     | 0.5590 (0.0496) |
| TLVQM        | 0.7729 (0.0242)  | **0.7988 (0.0365)**     | 0.6693 (0.0306) |
| VIDEVAL      | **0.7832 (0.0216)** | **0.7522 (0.0390)**  | **0.7787 (0.0254)** |


The median PLCC (std PLCC) of 100 repititions of different methods on different UGC-VQA datasets.

|    Methods   | KoNViD-1k             | LIVE-VQC             | YouTube-UGC         |
|:------------:|:---------------------:|:--------------------:|:-------------------:|
| BRISQUE      | 0.6576 (0.0342)  | 0.6380 (0.0632)     | 0.3952 (0.0486) |
| NIQE         | 0.5530 (0.0337)  | 0.6286 (0.0512)     | 0.2776 (0.0431) |
| IL-NIQE      | 0.5400 (0.0337)  | 0.5437 (0.0707)     | 0.3302 (0.0579) |
| GM-LOG       | 0.6636 (0.0315)  | 0.6212 (0.0636)     | 0.3920 (0.0549) |
| HIGRADE      | 0.7269 (0.0287)  | 0.6332 (0.0652)     | **0.7216 (0.0334)** |
| FRIQUEE      | 0.7482 (0.0257)  | 0.7000 (0.0587)     | **0.7571 (0.0324)** |
| CORNIA       | 0.7135 (0.0236)  | 0.7183 (0.0420)     | 0.6057 (0.0399) |
| HOSA         | 0.7664 (0.0207)  | **0.7414 (0.0410)**     | 0.6047 (0.0347) |
| VGG-19       | **0.7845 (0.0246)**  | 0.7160 (0.0481) | 0.6997 (0.0281) |
| ResNet-50    | **0.8104 (0.0229)**  | 0.7205 (0.0434) | 0.7097 (0.0276) |
| VIIDEO       | 0.3002 (0.0539)  | 0.2146 (0.0903)     | 0.1534 (0.0498) |
| V-BLIINDS    | 0.7037 (0.0301)   | 0.7178 (0.0500)     | 0.5551 (0.0465) |
| TLVQM        | 0.7688 (0.0238)  | **0.8025 (0.0360)**     | 0.6590 (0.0302) |
| VIDEVAL      | **0.7803 (0.0223)** | **0.7514 (0.0420)**  | **0.7733 (0.0257)** |

## Citation

If you use this code for your research, please cite our papers.

```
@article{tu2020ugc,
  title={UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content},
  author={Tu, Zhengzhong and Wang, Yilin and Birkbeck, Neil and Adsumilli, Balu and Bovik, Alan C},
  journal={arXiv preprint arXiv:2005.14354},
  year={2020}
}
```

## Contact

Zhengzhong Tu, `zhengzhong.tu@utexas.edu`