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


def make_class_items(names, colors=None):
    class_items = []
    for id, name in enumerate(names, 1):
        class_item = MachineLearning.ClassItem(id=id, name=name)
        if colors:
            class_item.color = colors[id-1]
        class_items.append(class_item)
    return class_items


def make_model_config(names, task, backend=None, colors=None):
    model_config = MachineLearning()
    model_config.task = task
    class_items = make_class_items(names, colors=colors)
    model_config.class_items.extend(class_items)

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


def make_classification_geojson_label_store(
        uri,
        cell_size,
        ioa_thresh=0.5,
        use_intersection_over_cell=False,
        pick_min_class_id=True,
        background_class_id=None,
        infer_cells=True):
    label_store = LabelStore()
    label_store.classification_geojson_file.uri = uri
    options = label_store.classification_geojson_file.options
    options.ioa_thresh = ioa_thresh
    options.use_intersection_over_cell = use_intersection_over_cell
    options.pick_min_class_id = pick_min_class_id
    if background_class_id is not None:
        options.background_class_id = background_class_id
    options.cell_size = cell_size
    options.infer_cells = infer_cells
    return label_store


def format_strings(strings, format_dict):
    return [str.format(**format_dict) for str in strings]


def make_classification_geotiff_geojson_scene(
        id,
        raster_uris,
        stats_uri,
        cell_size,
        ground_truth_labels_uri=None,
        prediction_base_uri=None,
        channel_order=[0, 1, 2],
        ioa_thresh=0.5,
        use_intersection_over_cell=False,
        pick_min_class_id=True,
        background_class_id=None,
        infer_cells=True):
    scene = Scene()
    scene.id = id
    scene.raster_source.MergeFrom(make_geotiff_raster_source(
        raster_uris, stats_uri, channel_order=channel_order))

    if ground_truth_labels_uri:
        scene.ground_truth_label_store.MergeFrom(
            make_classification_geojson_label_store(
                ground_truth_labels_uri,
                cell_size,
                ioa_thresh=ioa_thresh,
                use_intersection_over_cell=use_intersection_over_cell,
                pick_min_class_id=pick_min_class_id,
                background_class_id=background_class_id,
                infer_cells=infer_cells))

    if prediction_base_uri:
        predictions_uri = join(prediction_base_uri, str(id) + '.json')
        scene.prediction_label_store.MergeFrom(
            make_classification_geojson_label_store(
                predictions_uri,
                cell_size,
                ioa_thresh=ioa_thresh,
                use_intersection_over_cell=use_intersection_over_cell,
                pick_min_class_id=pick_min_class_id,
                background_class_id=background_class_id,
                infer_cells=infer_cells))

    return scene


def make_compute_stats(raster_sources, stats_uri):
    config = ComputeRasterStatsConfig()
    # XXX need to make builder ignore raster_transformer
    config.raster_sources.MergeFrom(raster_sources)
    config.stats_uri = stats_uri
    return config


def make_make_chips(
        train_scenes,
        validation_scenes,
        model_config,
        output_uri,
        chip_size=300,
        debug=True):
    config = MakeTrainingChipsConfig()
    config.train_scenes.MergeFrom(train_scenes)
    config.validation_scenes.MergeFrom(validation_scenes)
    config.machine_learning.MergeFrom(model_config)
    config.options.chip_size = chip_size
    config.options.debug = debug
    config.options.output_uri = output_uri
    return config


def make_train(model_config, training_data_uri, output_uri):
    config = TrainConfig()
    config.machine_learning.MergeFrom(model_config)
    config.options.training_data_uri = training_data_uri
    config.options.output_uri = output_uri
    return config


def make_predict(
        model_config,
        scenes,
        chip_size,
        model_uri,
        prediction_package_uri,
        debug,
        debug_uri):
    config = PredictConfig()
    config.machine_learning.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.debug_uri = debug_uri
    config.options.chip_size = chip_size
    config.options.model_uri = model_uri
    config.options.prediction_package_uri = \
        prediction_package_uri
    return config


def make_eval(
        model_config,
        scenes,
        output_uri,
        debug):
    config = EvalConfig()
    config.machine_learning.MergeFrom(model_config)
    config.scenes.MergeFrom(scenes)
    config.options.debug = debug
    config.options.output_uri = output_uri
    return config
