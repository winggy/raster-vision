from os.path import join

from rastervision.workflows.config_utils import (
    make_classification_geotiff_geojson_scene, make_compute_stats,
    make_make_chips, make_train, make_predict, make_eval)
from rastervision.utils.files import save_json_config
from rastervision.protos.chain_workflow_pb2 import ChainWorkflowConfig

COMPUTE_STATS = 'compute-stats'
MAKE_CHIPS = 'make-chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'


class ChainWorkflowPaths():
    def __init__(self, base_uri, compute_stats_uri=None, make_chips_uri=None,
                 train_uri=None, predict_uri=None, eval_uri=None,
                 workflow_uri=None):
        self.compute_stats_uri = compute_stats_uri
        if not compute_stats_uri:
            self.compute_stats_uri = join(base_uri, COMPUTE_STATS)
        self.stats_uri = join(self.compute_stats_uri, 'stats.json')

        if not make_chips_uri:
            self.make_chips_uri = join(base_uri, MAKE_CHIPS)

        if not train_uri:
            self.train_uri = join(base_uri, TRAIN)
        self.model_uri = join(self.train_uri, 'model')

        if not predict_uri:
            self.predict_uri = join(base_uri, PREDICT)
        self.prediction_package_uri = join(
            self.predict_uri, 'prediction-package.zip')

        if not eval_uri:
            self.eval_uri = join(base_uri, EVAL)
        self.metrics_uri = join(self.eval_uri, 'metrics.json')

        if not workflow_uri:
            self.workflow_uri = join(base_uri, 'workflow.json')

    def get_config_uri(self, uri):
        return join(uri, 'config.json')

    def get_debug_uri(self, uri):
        return join(uri, 'debug')


class ChainWorkflowSceneGenerator():
    def __init__(self, paths, chip_size=300):
        self.paths = paths
        self.chip_size = chip_size

    def make_classification_geotiff_geojson_scene(
            self,
            id,
            raster_uris,
            ground_truth_labels_uri=None,
            channel_order=[0, 1, 2],
            ioa_thresh=0.5,
            use_intersection_over_cell=False,
            pick_min_class_id=True,
            background_class_id=None,
            infer_cells=True):
        return make_classification_geotiff_geojson_scene(
            id,
            raster_uris,
            self.paths.stats_uri,
            self.chip_size,
            ground_truth_labels_uri=ground_truth_labels_uri,
            prediction_base_uri=self.paths.predict_uri,
            channel_order=channel_order,
            ioa_thresh=ioa_thresh,
            use_intersection_over_cell=use_intersection_over_cell,
            pick_min_class_id=pick_min_class_id,
            background_class_id=background_class_id,
            infer_cells=infer_cells)


class ChainWorkflow():
    def __init__(
            self,
            paths,
            model_config,
            train_scenes,
            validation_scenes,
            chip_size=300,
            debug=True):
        self.paths = paths
        self.model_config = model_config
        self.train_scenes = train_scenes
        self.validation_scenes = validation_scenes
        self.chip_size = chip_size
        self.debug = debug

    def make_compute_stats(self):
        scenes = self.train_scenes + self.validation_scenes
        raster_sources = [scene.raster_source for scene in scenes]
        return make_compute_stats(raster_sources, self.paths.stats_uri)

    def make_make_chips(self):
        return make_make_chips(
            self.train_scenes,
            self.validation_scenes,
            self.model_config,
            self.paths.make_chips_uri,
            chip_size=self.chip_size,
            debug=self.debug)

    def make_train(self):
        return make_train(
            self.model_config,
            self.paths.make_chips_uri,
            self.paths.train_uri)

    def make_predict(self):
        return make_predict(
            self.model_config,
            self.validation_scenes,
            self.chip_size,
            self.paths.model_uri,
            self.paths.prediction_package_uri,
            self.debug,
            self.paths.get_debug_uri(self.paths.predict_uri))

    def make_eval(self):
        scenes = [
            scene for scene in self.validation_scenes
            if scene.HasField('ground_truth_label_store')
        ]
        return make_eval(
            self.model_config,
            scenes,
            self.paths.metrics_uri,
            self.debug)

    def save_config(self):
        config = ChainWorkflowConfig()

        config.model_config.MergeFrom(self.model_config)
        config.chip_size = self.chip_size
        config.train_scenes.extend(self.train_scenes)
        config.validation_scenes.extend(self.validation_scenes)

        config.compute_stats_uri = self.paths.compute_stats_uri
        config.make_chips_uri = self.paths.make_chips_uri
        config.train_uri = self.paths.train_uri
        config.predict_uri = self.paths.predict_uri
        config.eval_uri = self.paths.eval_uri

        config.compute_stats.MergeFrom(self.make_compute_stats())
        config.make_chips.MergeFrom(self.make_make_chips())
        config.train.MergeFrom(self.make_train())
        config.predict.MergeFrom(self.make_predict())
        config.eval.MergeFrom(self.make_eval())

        save_json_config(config, self.paths.workflow_uri)
        print('Wrote workflow config to: ' + self.paths.workflow_uri)
