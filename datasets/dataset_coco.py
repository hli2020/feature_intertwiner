import os
import shutil
import urllib.request
import zipfile
from datasets.eval.PythonAPI.pycocotools.coco import COCO
from datasets.eval.PythonAPI.pycocotools import mask as maskUtils
import skimage.color
import skimage.io
from lib.layers import *
import torch
import tools.image_utils as utils
import torch.utils.data


class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    # def append_data(self, class_info, image_info):
    #     self.external_to_class_id = {}
    #     for i, c in enumerate(self.class_info):
    #         for ds, id in c["map"]:
    #             self.external_to_class_id[ds + str(id)] = i
    #
    #     # Map external image IDs to internal ones.
    #     self.external_to_image_id = {}
    #     for i, info in enumerate(self.image_info):
    #         self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    # def source_image_link(self, image_id):
    #     """Returns the path or URL to the image.
    #     Override this to return a URL to the image if it's availble online for easy
    #     debugging.
    #     """
    #     return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_coco(self, dataset_dir, subset, year='2014', class_ids=None, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir:    The root directory of the COCO dataset.
        subset:         What to load (train, val, minival, valminusminival)
        year:           What dataset year to load (2014, 2017) as a string, not an integer
        class_ids:      If provided, only loads images that have the given classes.
        class_map:      TODO: Not implemented yet.
                            Supports mapping classes from different datasets to the same class ID.
        return_coco:    If True, returns the COCO object.
        auto_download:  Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations datasets paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            # return super(COCODataset, self).load_mask(image_id)
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
            return mask, class_ids

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            # Call super class to return an empty mask
            # return super(CocoDataset, self).load_mask(image_id)
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
        return mask, class_ids

    # def image_reference(self, image_id):
    #     """Return a link to the image in the COCO Website."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "coco":
    #         return "http://cocodataset.org/#explore?id={}".format(info["id"])
    #     else:
    #         # super(CocoDataset, self).image_reference(image_id)
    #         return ""

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


class COCODataset(torch.utils.data.Dataset):
    ONLY_ONCE = True

    def __init__(self, config, augment=True):
        """A generator that returns images and corresponding target class ids,
            bounding box deltas, and masks.

            dataset:    The Dataset object to pick datasets from
            config:     The model config object
            shuffle:    If True, shuffles the samples before every epoch
            augment:    If True, applies image augmentation to images (currently only horizontal flips are supported)

            Returns a Python generator. Upon calling next() on it, the
            generator returns two lists, inputs and outputs. The containers
            of the lists differs depending on the received arguments:
            inputs list:
            - images:       [batch, H, W, C]
            - image_metas:  [batch, size of image meta]
            - rpn_match:    [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
            - rpn_bbox:     [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
            - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
            - gt_boxes:     [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
            - gt_masks:     [batch, height, width, MAX_GT_INSTANCES]. The height and width
                                are those of the image unless use_mini_mask is True, in which
                                case they are defined in MINI_MASK_SHAPE.

            outputs list: Usually empty in regular training. But if detection_targets
                is True then the outputs list contains target class_ids, bbox deltas, and masks.
            """
        self.dataset = Dataset()
        # self.image_ids = np.copy(self.dataset.image_ids)
        self.config = config
        self.augment = augment

    def __getitem__(self, image_index):

        # Get GT bounding boxes and masks for image.
        image_id = self.dataset.image_ids[image_index]

        image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
            utils.load_image_and_gt(self.dataset, self.config, image_id, augment=self.augment,
                                    use_mini_mask=self.config.MRCNN.USE_MINI_MASK)

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        # UPDATE: never run into this case
        if not np.any(gt_class_ids > 0):
            return None

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.DATA.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.DATA.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        image = image.astype(np.float32) - self.config.DATA.MEAN_PIXEL
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        gt_masks = gt_masks.astype(int).transpose(2, 0, 1)

        return image, gt_class_ids, gt_boxes, gt_masks, image_metas

    def __len__(self):
        return self.dataset.image_ids.shape[0]


def detection_collate(batch):
    """Custom collate function for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    """
    imgs = []
    imgs_metas = []
    # rpn_match, rpn_bbox = [], []
    gt_class_ids, gt_boxes, gt_masks = [], [], []
    for sample in batch:
        imgs.append(sample[0])
        # rpn_match.append(sample[2])
        # rpn_bbox.append(sample[3])
        gt_class_ids.append(sample[1])
        gt_boxes.append(sample[2])
        gt_masks.append(sample[3])
        imgs_metas.append(sample[4])

    return torch.stack(imgs, 0), \
           gt_class_ids, gt_boxes, gt_masks, \
           torch.stack(imgs_metas, 0)


def get_data(config):

    DATASET = config.DATASET

    # validation data
    dset_val = COCODataset(config)
    print('VAL:: load minival')
    val_coco_api = dset_val.dataset.load_coco(DATASET.PATH, "minival", year=DATASET.YEAR)
    dset_val.dataset.prepare()

    # train data
    if not config.CTRL.DEBUG and config.CTRL.PHASE == 'train' and not config.CTRL.QUICK_VERIFY:
        dset_train = COCODataset(config)
        print('TRAIN:: load train')
        dset_train.dataset.load_coco(DATASET.PATH, "train", year=DATASET.YEAR)
        print('TRAIN:: load val_minus_minival')
        dset_train.dataset.load_coco(DATASET.PATH, "valminusminival", year=DATASET.YEAR)
        dset_train.dataset.prepare()
    else:
        # if QUICK_VERIFY=True, use this
        dset_train = dset_val

    train_generator = None if config.CTRL.PHASE == 'inference' else \
        torch.utils.data.DataLoader(dset_train, batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=True, num_workers=config.DATA.LOADER_WORKER_NUM,
                                    collate_fn=detection_collate)
                                    # collate_fn=detection_collate, drop_last=True, pin_memory=True)

    return train_generator, dset_val, val_coco_api

