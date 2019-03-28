
## Installation
1. Clone this repository.

        git clone --recursive https://github.com/hli2020/feature_intertwiner.git

    
2. We use functions from other repositories that need to be build with the right `--arch` option for cuda support.
The functions are Non-Maximum Suppression from ruotianluo's [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn)
repository and longcw's [RoiAlign](https://github.com/longcw/RoIAlign.pytorch) and RoiPool. Thanks to them!

        sh setup.sh

3. As we use the [COCO dataset](http://cocodataset.org/#home),
install the [Python COCO API](https://github.com/cocodataset/cocoapi) and
create a symlink.

        ln -s /path/to/coco datasets/coco

4. Download the pretrained models on COCO and ImageNet from
[Google Drive](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).


### Minor installation problems

- You might encounter some warnings on interpolation; please 
comment the source files if necessary:

    ``~/anaconda3/lib/python3.6/site-packages/scipy/ndimage/interpolation.py:616``
 
 
- Install the `future` package via conda:

    ``conda install -c anaconda future``


### Install visdom (optional)
**No need** to install visdom on local machine. 

- If you use visdom, follow instructions below (on a **remote** server):
    ```shell
    git clone --recursive https://github.com/facebookresearch/visdom.git
    cd visdom
    pip install -e .
    ```
    Then activate `visdom` in a remote server like this: ``python -m visdom.server -port=$PORT_ID``, where `$PORT_ID` 
    is the port number like 2042, 8089, etc.
    
- On the local machine, ssh the server via `ssh your_regular_connection_path -L $PORT_ID:127.0.0.1:$PORT_ID`.
    
- You are all set! Now you can browse the remote work on-the-fly via: `http://localhost:$PORT_ID` in your browser. 
    






