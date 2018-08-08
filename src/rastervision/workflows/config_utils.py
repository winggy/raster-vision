from os.path import join

from rastervision.protos.machine_learning_pb2 import MachineLearning
from rastervision.protos.scene_pb2 import Scene
from rastervision.protos.raster_source_pb2 import RasterSource
from rastervision.protos.label_store_pb2 import LabelStore
from rastervision.protos.compute_raster_stats_pb2 import ComputeRasterStatsConfig
from rastervision.protos.make_training_chips_pb2 import MakeTrainingChipsConfig
from rastervision.protos.train_pb2 import TrainConfig
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.protos.eval_pb2 import EvalConfig


OD = MachineLearning.Task.Value('OBJECT_DETECTION')
CL = MachineLearning.Task.Value('CLASSIFICATION')
TFOD = \
    MachineLearning.Backend.Value('TF_OBJECT_DETECTION_API')
KERAS = MachineLearning.Backend.Value('KERAS_CLASSIFICATION')

BACKGROUND = 'background'

COMPUTE_STATS = 'compute-stats'
MAKE_CHIPS = 'make-chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_COMMANDS = [COMPUTE_STATS, MAKE_CHIPS, TRAIN, PREDICT, EVAL]


def make_stats_uri(compute_stats_uri):
    return join(compute_stats_uri, 'stats.json')


def make_model_uri(train_uri):
    return join(train_uri, 'model')


def make_prediction_package_uri(predict_uri):
    return join(predict_uri, 'prediction-package.zip')


def make_metrics_uri(eval_uri):
    return join(eval_uri, 'metrics.json')


def make_debug_uri(uri):
    return join(uri, 'debug')


def make_command_config_uri(command_uri, command):
    return join(command_uri, command + '-config.json')


def make_class_items(names, colors=None):
    class_items = []
    for id, name in enumerate(names, 1):
        class_item = MachineLearning.ClassItem(id=id, name=name)
        if colors:
            class_item.color = colors[id-1]
        class_items.append(class_item)
    return class_items


def make_model_config(class_names, task, backend=None, colors=None):
    model_config = MachineLearning()
    model_config.task = task

    class_items = make_class_items(class_names, colors=colors)
    model_config.class_items.MergeFrom(class_items)

    if not backend:
        if task == OD:
            backend = TFOD
        elif task == CL:
            backend = KERAS
    model_config.backend = backend
    return model_config


def make_geotiff_raster_source(
        raster_uris,
        stats_uri,
        channel_order=[0, 1, 2]):
    raster_source = RasterSource()
    raster_source.geotiff_files.uris.extend(raster_uris)
    raster_source.raster_transformer.stats_uri = stats_uri
    raster_source.raster_transformer.channel_order.extend(channel_order)
    return raster_source


class ClassificationGeoJSONOptions():
    def __init__(
            self,
            cell_size=300,
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            background_class_id=None,
            pick_min_class_id=True,
            infer_cells=True):
        self.cell_size = cell_size
        self.ioa_thresh = ioa_thresh
        self.use_intersection_over_cell = use_intersection_over_cell
        self.background_class_id = background_class_id
        self.pick_min_class_id = pick_min_class_id
        self.infer_cells = infer_cells


class ObjectDetectionGeoJSONOptions():
    pass


def make_geojson_label_store(uri, task_options):
    label_store = LabelStore()

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionGeoJSONOptions()
        elif task_options == CL:
            task_options = ClassificationGeoJSONOptions()

    if task_options is not None:
        if type(task_options) is ObjectDetectionGeoJSONOptions:
            label_store.object_detection_geojson_file.uri = uri
        elif type(task_options) is ClassificationGeoJSONOptions:
            label_store.classification_geojson_file.uri = uri
            options = label_store.classification_geojson_file.options
            options.ioa_thresh = task_options.ioa_thresh
            options.use_intersection_over_cell = \
                task_options.use_intersection_over_cell
            options.pick_min_class_id = task_options.pick_min_class_id
            if task_options.background_class_id is not None:
                options.background_class_id = task_options.background_class_id
            options.cell_size = task_options.cell_size
            options.infer_cells = task_options.infer_cells
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))

    return label_store


def format_strings(strings, format_dict):
    return [str.format(**format_dict) for str in strings]


def make_geotiff_geojson_scene(
        id,
        raster_uris,
        stats_uri,
        task_options,
        ground_truth_labels_uri=None,
        prediction_base_uri=None,
        channel_order=[0, 1, 2]):
    scene = Scene()
    scene.id = id
    scene.raster_source.MergeFrom(make_geotiff_raster_source(
        raster_uris, stats_uri, channel_order=channel_order))

    if ground_truth_labels_uri:
        scene.ground_truth_label_store.MergeFrom(
            make_geojson_label_store(
                ground_truth_labels_uri,
                task_options))

    if prediction_base_uri:
        predictions_uri = join(prediction_base_uri, str(id) + '.json')
        scene.prediction_label_store.MergeFrom(
            make_geojson_label_store(
                predictions_uri,
                task_options))

    return scene


def make_compute_stats(raster_sources, output_uri):
    config = ComputeRasterStatsConfig()
    # XXX need to make builder ignore raster_transformer
    config.raster_sources.MergeFrom(raster_sources)
    config.stats_uri = make_stats_uri(output_uri)
    return config


class ObjectDetectionMakeChipsOptions():
    def __init__(
            self,
            neg_ratio=1.0,
            ioa_thresh=0.8,
            window_method='chip',
            label_buffer=0.0):
        self.neg_ratio = neg_ratio
        self.ioa_thresh = ioa_thresh
        self.window_method = window_method
        self.label_buffer = label_buffer


def make_make_chips(
        train_scenes,
        validation_scenes,
        model_config,
        output_uri,
        task_options,
        chip_size=300,
        debug=True):
    config = MakeTrainingChipsConfig()
    config.train_scenes.MergeFrom(train_scenes)
    config.validation_scenes.MergeFrom(validation_scenes)
    config.machine_learning.MergeFrom(model_config)
    config.options.chip_size = chip_size
    config.options.debug = debug
    config.options.output_uri = output_uri

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionMakeChipsOptions()
        elif task_options == CL:
            task_options = None
        else:
            raise ValueError('Unknown task: ' + task_options)

    if task_options is not None:
        if type(task_options) is ObjectDetectionMakeChipsOptions:
            config.options.object_detection_options.neg_ratio = \
                task_options.neg_ratio
            config.options.object_detection_options.ioa_thresh = \
                task_options.ioa_thresh
            config.options.object_detection_options.window_method = \
                task_options.window_method
            config.options.object_detection_options.label_buffer = \
                task_options.label_buffer
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))

    return config


def make_train(
        model_config,
        backend_config_uri,
        make_chips_uri,
        output_uri,
        pretrained_model_uri,
        sync_interval=600):
    config = TrainConfig()
    config.machine_learning.MergeFrom(model_config)
    config.options.backend_config_uri = backend_config_uri
    config.options.training_data_uri = make_chips_uri
    config.options.output_uri = output_uri
    config.options.pretrained_model_uri = pretrained_model_uri
    config.options.sync_interval = sync_interval
    return config


class ObjectDetectionPredictOptions():
    def __init__(self, merge_thresh=0.5, score_thresh=0.5):
        self.merge_thresh = merge_thresh
        self.score_thresh = score_thresh


def make_predict(
        model_config,
        scenes,
        chip_size,
        train_uri,
        output_uri,
        task_options,
        debug=True):
    config = PredictConfig()
    config.machine_learning.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.debug_uri = make_debug_uri(output_uri)
    config.options.chip_size = chip_size
    config.options.model_uri = make_model_uri(train_uri)
    config.options.prediction_package_uri = \
        make_prediction_package_uri(output_uri)

    if type(task_options) is int:
        if task_options == OD:
            task_options = ObjectDetectionPredictOptions()
        elif task_options == CL:
            task_options = None
        else:
            raise ValueError('Unknown task: ' + task_options)

    if task_options is not None:
        if type(task_options) is ObjectDetectionPredictOptions:
            config.options.object_detection_options.merge_thresh = \
                task_options.merge_thresh
            config.options.object_detection_options.score_thresh = \
                task_options.score_thresh
        else:
            raise ValueError('Unknown type of task_options: ' +
                             str(type(task_options)))
    return config


def make_eval(
        model_config,
        scenes,
        output_uri,
        debug=True):
    config = EvalConfig()
    config.machine_learning.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.output_uri = make_metrics_uri(output_uri)
    return config
