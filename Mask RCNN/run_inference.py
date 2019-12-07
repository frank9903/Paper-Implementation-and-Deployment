import os
import sys
import skimage.io
import tensorflow as tf
from mrcnn.utils import download_file_from_google_drive
from mrcnn.model_utils import detect
from mrcnn import visualize, inference
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Root directory of the project
ROOT_DIR = os.path.abspath(".")
from mrcnn.config import Config
from keras.backend import clear_session

class MyConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "default"

    IMAGES_PER_GPU = 1

    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 5 + 1

config = MyConfig()

# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane']

def load_model():
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_0160.h5")
    # Download COCO trained weights if needed
    if not os.path.exists(COCO_MODEL_PATH):
        download_file_from_google_drive("14u3dr4vSEzTwt0ircQxqGI7sFsQ-55So", COCO_MODEL_PATH)

    # Create inference model object
    model = inference.build_inference_model(config)
    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model

global inference_model
inference_model = load_model()
global graph
graph = tf.get_default_graph()

def run(image_url, save_path=None):
    image = skimage.io.imread(image_url)

    # Run detection
    with graph.as_default():
        results = detect(inference_model, [image], config, threshold=0.95)
    r = results[0]

    if not save_path:
        save_path = ROOT_DIR+"/result.png"
    # Visualize results
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], save_path=save_path)
    return save_path

######################################## TEST ########################################
# Directory of images to run detection on
# IMAGE_DIR = os.path.join(ROOT_DIR, "test_images")
# # test image
# # image = skimage.io.imread(os.path.join(IMAGE_DIR, "1.jpg"))
# image = skimage.io.imread("/Users/shuheng/Documents/iCloud/2019F/IOS_UPLOAD_TO_DJANGO_DEMO-master/myproject/media/documents/2019/12/04/photo_zrIIXvV.jpg")
# run(image)
# run("/Users/shuheng/Documents/iCloud/2019F/IOS_UPLOAD_TO_DJANGO_DEMO-master/myproject/media/documents/2019/12/04/photo_zrIIXvV.jpg")
