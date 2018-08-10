from os.path import join, dirname

from google.protobuf import text_format
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig

from rastervision.workflows.config_utils import (
    TFOD, KERAS)
from rastervision.utils.files import file_to_str, str_to_file

MOBILENET = 'mobilenet'
RESNET50 = 'resnet50'
LOCAL = 'local'
REMOTE = 'remote'

pretrained_model_uris = {
    TFOD: {
        MOBILENET: {
            LOCAL: '/opt/data/lf-dev/pretrained-models/tf-object-detection-api/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',
            REMOTE: 's3://raster-vision-lf-dev/pretrained-models/tf-object-detection-api/ssd_mobilenet_v1_coco_2017_11_17.tar.gz'
        }
    },
    KERAS: {
        RESNET50: {
            LOCAL: None,
            REMOTE: None
        }
    }
}

default_model_types = {
    TFOD: MOBILENET,
    KERAS: RESNET50
}


def make_backend_config_uri(base_uri):
    return join(base_uri, 'backend_config.txt')


def get_pretrained_model_uri(backend, model_type=None, is_remote=False):
    if model_type is None:
        model_type = default_model_types[backend]
    environment = REMOTE if is_remote else LOCAL
    return pretrained_model_uris[backend][model_type][environment]


def save_mobilenet_config(backend_config_uri, num_classes, batch_size=8,
                          num_steps=50000):
    sample_path = join(
        dirname(__file__), 'samples', 'backend-configs',
        'tf-object-detection-api', 'mobilenet.config')
    config = text_format.Parse(
        file_to_str(sample_path), TrainEvalPipelineConfig())
    config.model.ssd.num_classes = num_classes
    config.train_config.batch_size = batch_size
    config.train_config.num_steps = num_steps
    str_to_file(text_format.MessageToString(config), backend_config_uri)


def save_backend_config(backend_config_uri, backend, num_classes,
                        model_type=None, batch_size=None, num_steps=None):
    if model_type is None:
        model_type = default_model_types[backend]

    if model_type == MOBILENET:
        if batch_size is None:
            batch_size = 8
        if num_steps is None:
            num_steps = 50000
        save_mobilenet_config(
            backend_config_uri, num_classes, batch_size=batch_size,
            num_steps=num_steps)
