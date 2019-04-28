## Feature Intertwiner for Object Detection


A PyTorch implementation of our paper published at ICLR 2019.

By [Hongyang Li](http://www.ee.cuhk.edu.hk/~yangli/), 
[Bo Dai](http://daibo.info/),
[Shaoshuai Shi](https://scholar.google.com.hk/citations?user=DC9wzBgAAAAJ&hl=en), 
Wanli Ouyang, and Xiaogang Wang.

**Paper:** 
[[arXiv]](https://arxiv.org/abs/1903.11851)
[[Openreview]](https://openreview.net/forum?id=SyxZJn05YX) 

**A 50-min talk** presented at GTC 2019:
[[GTC Video]](https://on-demand.gputechconf.com/gtc/2019/video/_/S9551/) 
[[GTC Slides]](https://docs.google.com/presentation/d/12Syg5OXD6nGwtG_nwmoQ4kqX5GtJ-5pJ1OuVY53FqB0/edit?usp=sharing)

### Overview

Our assumption is that semantic features for one category should be the same as shown
in (a) below. Due to the inferior up-sampling design in RoI operation, shown in 
(b), the reliable set (green) *could* guide the feature learning of the less
reliable set (blue).

![](assets/motivation_new.png | width=100)

<img src="assets/motivation_new.png" width="200">


Here comes the proposed feature intertwiner:

![](assets/intertwiner.png | width=100)

![](https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png =250x250)

<img src="https://cloud.githubusercontent.com/assets/yourgif.gif" width="400" height="790">


- PyTorch `0.3` 
- Code/framework based on Mask-RCNN.
- Datasets: COCO and Pascal VOC (not in this repo)

### How to run

Follow instructions in [`INSTALL.md`](INSTALL.md) to 
set up datasets, symlinks, compilation, etc.

##### To train
```bash
sh script/base_4gpu    105/meta_105_quick_1   0,2,5,7   # gpu ids
```
or execute `python main.py`. The configurations are stored 
in the `configs` folder.

##### To test

Change the flag `--phase` in `main.py` to `inference`. 

##### Performance

TODO.

### Adapting Feature Intertwiner to your own task

TODO.

### Citation
Please cite in the following manner if you find it useful in your research:
```
@inproceedings{li2019_internet,
  title = {{Feature Intertwiner for Object Detection}},
  author = {Hongyang Li and Bo Dai and Shaoshuai Shi and Wanli Ouyanbg and Xiaogang Wang},
  booktitle = {ICLR},
  year = {2019}
}
```


