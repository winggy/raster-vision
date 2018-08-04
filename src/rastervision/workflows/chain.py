from os.path import join, isfile
import copy
from urllib.parse import urlparse
import subprocess

import click
from google.protobuf.descriptor import FieldDescriptor
import boto3
import botocore

from rastervision.protos.chain_workflow_pb2 import ChainWorkflowConfig
from rastervision.protos.compute_raster_stats_pb2 import (
    ComputeRasterStatsConfig)
from rastervision.protos.make_training_chips_pb2 import (
    MakeTrainingChipsConfig)
from rastervision.protos.train_pb2 import TrainConfig
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.protos.eval_pb2 import EvalConfig
from rastervision.protos.label_store_pb2 import (
    LabelStore as LabelStoreConfig, ObjectDetectionGeoJSONFile as
    ObjectDetectionGeoJSONFileConfig, ClassificationGeoJSONFile as
    ClassificationGeoJSONFileConfig)

from rastervision.utils.files import (load_json_config, save_json_config,
                                      file_to_str, str_to_file)
from rastervision.utils.batch import _batch_submit
from rastervision import run

COMPUTE_RASTER_STATS = 'compute-raster-stats'
MAKE_TRAINING_CHIPS = 'make-training-chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_TASKS = [COMPUTE_RASTER_STATS, MAKE_TRAINING_CHIPS, TRAIN, PREDICT, EVAL]

validated_uri_fields = set(
    [('rv.protos.ObjectDetectionGeoJSONFile',
      'uri'), ('rv.protos.ClassificationGeoJSONFile', 'uri'),
     ('rv.protos.GeoTiffFiles', 'uris'), ('rv.protos.ImageFile', 'uri'),
     ('rv.protos.TrainConfig.Options',
      'backend_config_uri'), ('rv.protos.TrainConfig.Options',
                              'pretrained_model_uri')])

s3 = boto3.resource('s3')


def make_command(command, config_uri):
    return 'python -m rastervision.run {} {}'.format(command, config_uri)


def get_config_uri(prefix_uri):
    return join(prefix_uri, 'config.json')


def is_branch_valid(branch):
    ls_branch_command = [
        'git', 'ls-remote', '--heads',
        'https://github.com/azavea/raster-vision.git', branch
    ]

    if not subprocess.run(ls_branch_command, stdout=subprocess.PIPE).stdout:
        print('Error: remote branch {} does not exist'.format(branch))
        return False
    return True


def is_uri_valid(uri):
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme == 's3':
        try:
            s3.Object(parsed_uri.netloc, parsed_uri.path[1:]).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print('Error: URI cannot be found: {}'.format(uri))
                print(e)
                return False
    else:
        if not isfile(uri):
            print('Error: URI cannot be found: {}'.format(uri))
            return False

    return True


def is_validated_uri_field(message_type, field_name):
    return (message_type, field_name) in validated_uri_fields


def is_config_valid(config):
    # If config is primitive, do nothing.
    if not hasattr(config, 'ListFields'):
        return True

    message_type = config.DESCRIPTOR.full_name

    is_valid = True
    for field_desc, field_val in config.ListFields():
        field_name = field_desc.name

        if is_validated_uri_field(message_type, field_name):
            if field_name.endswith('uri'):
                is_valid = is_uri_valid(field_val) and is_valid

            if field_name.endswith('uris'):
                for uri in field_val:
                    is_valid = is_uri_valid(uri) and is_valid

        # Recurse.
        if field_desc.label == FieldDescriptor.LABEL_REPEATED:
            for field_val_item in field_val:
                is_valid = \
                    is_config_valid(field_val_item) and is_valid
        else:
            is_valid = is_config_valid(field_val) and is_valid

    return is_valid


def apply_uri_map(config, uri_map):
    """Do parameter substitution on any URI fields."""

    def _apply_uri_map(config):
        # If config is primitive, do nothing.
        if not hasattr(config, 'ListFields'):
            return

        # For each field in message, update its value if the name ends with
        # uri or uris.
        for field_desc, field_val in config.ListFields():
            field_name = field_desc.name

            if field_name.endswith('uri'):
                new_uri = field_val.format(**uri_map)
                setattr(config, field_name, new_uri)

            if field_name.endswith('uris'):
                for ind, uri in enumerate(field_val):
                    new_uri = uri.format(**uri_map)
                    field_val[ind] = new_uri

            # Recurse.
            if field_desc.label == FieldDescriptor.LABEL_REPEATED:
                for field_val_item in field_val:
                    _apply_uri_map(field_val_item)
            else:
                _apply_uri_map(field_val)

    new_config = config.__deepcopy__()
    _apply_uri_map(new_config)
    return new_config


class ChainWorkflow(object):
    def __init__(self, workflow_uri, remote=False):
        self.workflow = load_json_config(workflow_uri, ChainWorkflowConfig())
        self.uri_map = (self.workflow.remote_uri_map
                        if remote else self.workflow.local_uri_map)
        self.update_output_uris()
        self.uri_parameter_substitution()

        is_valid = is_config_valid(self.workflow)
        if not is_valid:
            exit()

        self.update_raster_transformer()
        self.update_scenes()

    def update_output_uris(self):
        if self.workflow.nested_output:
            self.workflow.compute_raster_stats_uri = join(
                '{output_base}', '{}-{}'.format(
                    self.workflow.compute_raster_stats_uri,
                    'stats',
                ))
            self.workflow.make_training_chips_uri = join(
                self.workflow.compute_raster_stats_uri, '{}-{}'.format(
                    self.workflow.make_training_chips_uri,
                    'chips'))
            self.workflow.train_uri = join(
                self.workflow.make_training_chips_uri, '{}-{}'.format(
                    self.workflow.train_uri, TRAIN))
            self.workflow.predict_uri = join(
                self.workflow.train_uri, '{}-{}'.format(
                    self.workflow.predict_uri, PREDICT))
            self.workflow.eval_uri = join(
                self.workflow.predict_uri, '{}-{}'.format(
                    self.workflow.eval_uri, EVAL))

    def uri_parameter_substitution(self):
        # Hack to deal with fact that fields set by default values are not
        # recognized by the ListFields method, which is called in
        # apply_uri_map.
        self.workflow.compute_raster_stats_uri = \
            self.workflow.compute_raster_stats_uri
        self.workflow.make_training_chips_uri = \
            self.workflow.make_training_chips_uri
        self.workflow.train_uri = self.workflow.train_uri
        self.workflow.predict_uri = self.workflow.predict_uri
        self.workflow.eval_uri = self.workflow.eval_uri

        self.workflow = apply_uri_map(self.workflow, self.uri_map)

    def update_raster_transformer(self):
        stats_uri = join(self.workflow.compute_raster_stats_uri, 'stats.json')
        self.workflow.raster_transformer.stats_uri = stats_uri

    def update_label_store(self, label_store):
        """Set label store options based on label_store_template."""
        if not label_store:
            return

        label_store_template = self.workflow.label_store_template
        label_store_type = label_store_template.WhichOneof('label_store_type')

        if label_store_type == 'object_detection_geojson_file':
            template_options = \
                label_store_template.object_detection_geojson_file.options
            if not label_store.object_detection_geojson_file.HasField(
                    'options'):
                label_store.object_detection_geojson_file.options.MergeFrom(
                    template_options)
        elif label_store_type == 'classification_geojson_file':
            template_options = \
                label_store_template.classification_geojson_file.options
            if not label_store.classification_geojson_file.HasField('options'):
                label_store.classification_geojson_file.options.MergeFrom(
                    template_options)
        else:
            raise ValueError('Not sure how to update label_store of type {}'
                             .format(label_store_type))

    def make_prediction_label_store(self, scene):
        """Make prediction_label_store based on scene.id"""
        prediction_uri = join(self.workflow.predict_uri,
                              '{}.json'.format(scene.id))

        label_store_template = self.workflow.label_store_template
        label_store_type = label_store_template.WhichOneof('label_store_type')
        if label_store_type == 'object_detection_geojson_file':
            geojson_file = ObjectDetectionGeoJSONFileConfig(uri=prediction_uri)
            return LabelStoreConfig(object_detection_geojson_file=geojson_file)
        elif label_store_type == 'classification_geojson_file':
            geojson_file = ClassificationGeoJSONFileConfig(uri=prediction_uri)
            return LabelStoreConfig(classification_geojson_file=geojson_file)

    def update_scene(self, prefix, idx, scene):
        """Set id, raster_transformer, and options in label_stores."""
        if len(scene.id) < 1:
            scene.id = '{}-{}'.format(prefix, idx)
        scene.raster_source.raster_transformer.MergeFrom(
            self.workflow.raster_transformer)
        if scene.HasField('prediction_label_store'):
            self.update_label_store(scene.prediction_label_store)
        if scene.HasField('ground_truth_label_store'):
            self.update_label_store(scene.ground_truth_label_store)

    def update_scenes(self):
        """Fill in missing fields in all scenes."""
        for idx, scene in enumerate(self.workflow.train_scenes):
            self.update_scene('train', idx, scene)

        for idx, scene in enumerate(self.workflow.validation_scenes):
            scene.prediction_label_store.MergeFrom(
                self.make_prediction_label_store(scene))
            self.update_scene('validation', idx, scene)

        for idx, scene in enumerate(self.workflow.test_scenes):
            scene.prediction_label_store.MergeFrom(
                self.make_prediction_label_store(scene))
            self.update_scene('test', idx, scene)

    def get_compute_raster_stats_config(self):
        config = ComputeRasterStatsConfig()
        scenes = copy.deepcopy(self.workflow.train_scenes)
        scenes.extend(self.workflow.validation_scenes)
        scenes.extend(self.workflow.test_scenes)

        for scene in scenes:
            # Set the raster_transformer so its fields are null since
            # compute_raster_stats will generate stats_uri.
            raster_source = copy.deepcopy(scene.raster_source)
            raster_source.raster_transformer.stats_uri = ''
            config.raster_sources.extend([raster_source])
        config.stats_uri = self.workflow.raster_transformer.stats_uri
        config = apply_uri_map(config, self.uri_map)
        return config

    def get_make_training_chips_config(self):
        # If no validation scenes, use test scenes that contain
        # ground_truth_label_stores.
        config = MakeTrainingChipsConfig()
        config.train_scenes.MergeFrom(self.workflow.train_scenes)
        config.validation_scenes.MergeFrom(self.workflow.validation_scenes)
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.make_training_chips_options)
        config.options.chip_size = self.workflow.chip_size
        config.options.debug = self.workflow.debug
        config.options.output_uri = self.workflow.make_training_chips_uri

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_train_config(self):
        config = TrainConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.train_options)
        config.options.training_data_uri = \
            self.workflow.make_training_chips_uri
        config.options.output_uri = \
            self.workflow.train_uri

        # Copy backend config so that it is nested under model_uri. This way,
        # all config files and corresponding output of RV will be located next
        # to each other in the file system.
        backend_config_copy_uri = join(self.workflow.train_uri,
                                       'backend.config')
        backend_config_uri = config.options.backend_config_uri.format(
            **self.uri_map)
        backend_config_str = file_to_str(backend_config_uri)
        str_to_file(backend_config_str, backend_config_copy_uri)
        config.options.backend_config_uri = backend_config_copy_uri

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_predict_config(self):
        config = PredictConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.scenes.MergeFrom(self.workflow.validation_scenes)
        config.scenes.MergeFrom(self.workflow.test_scenes)
        config.options.MergeFrom(self.workflow.predict_options)
        config.options.debug = self.workflow.debug
        config.options.debug_uri = join(self.workflow.predict_uri, 'debug')
        config.options.chip_size = self.workflow.chip_size
        config.options.model_uri = join(self.workflow.train_uri, 'model')
        config.options.prediction_package_uri = join(self.workflow.predict_uri,
                                                     'predict-package.zip')

        config = apply_uri_map(config, self.uri_map)

        return config

    def get_eval_config(self):
        config = EvalConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        # Use test_scenes with ground_truth_label_stores as eval_scenes.
        eval_scenes = [
            scene for scene in self.workflow.test_scenes
            if scene.HasField('ground_truth_label_store')
        ]
        eval_scenes.extend(self.workflow.validation_scenes)
        config.scenes.MergeFrom(eval_scenes)
        config.options.MergeFrom(self.workflow.eval_options)
        config.options.debug = self.workflow.debug
        config.options.output_uri = join(self.workflow.eval_uri, 'eval.json')
        config = apply_uri_map(config, self.uri_map)
        return config

    def save_configs(self, tasks):
        print('Generating and saving config files:')

        if COMPUTE_RASTER_STATS in tasks:
            uri = get_config_uri(self.workflow.compute_raster_stats_uri)
            print(uri)
            save_json_config(self.get_compute_raster_stats_config(), uri)

        if MAKE_TRAINING_CHIPS in tasks:
            uri = get_config_uri(self.workflow.make_training_chips_uri)
            print(uri)
            save_json_config(self.get_make_training_chips_config(), uri)

        if TRAIN in tasks:
            uri = get_config_uri(self.workflow.train_uri)
            print(uri)
            save_json_config(self.get_train_config(), uri)

        if PREDICT in tasks:
            uri = get_config_uri(self.workflow.predict_uri)
            print(uri)
            save_json_config(self.get_predict_config(), uri)

        if EVAL in tasks:
            uri = get_config_uri(self.workflow.eval_uri)
            print(uri)
            save_json_config(self.get_eval_config(), uri)

    def remote_run(self, tasks, branch):
        if not is_branch_valid(branch):
            exit()

        # Run everything in GPU queue since Batch doesn't seem to
        # handle dependencies across different queues.
        parent_job_ids = []

        if COMPUTE_RASTER_STATS in tasks:
            command = make_command(
                COMPUTE_RASTER_STATS,
                get_config_uri(self.workflow.compute_raster_stats_uri))
            job_id = _batch_submit(branch, command, attempts=1, gpu=True)
            parent_job_ids = [job_id]

        if MAKE_TRAINING_CHIPS in tasks:
            command = make_command(
                MAKE_TRAINING_CHIPS,
                get_config_uri(self.workflow.make_training_chips_uri))
            job_id = _batch_submit(
                branch,
                command,
                attempts=1,
                gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if TRAIN in tasks:
            command = make_command(TRAIN,
                                   get_config_uri(self.workflow.train_uri))
            job_id = _batch_submit(
                branch,
                command,
                attempts=1,
                gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if PREDICT in tasks:
            command = make_command(PREDICT,
                                   get_config_uri(self.workflow.predict_uri))
            job_id = _batch_submit(
                branch,
                command,
                attempts=1,
                gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if EVAL in tasks:
            command = make_command(EVAL,
                                   get_config_uri(self.workflow.eval_uri))
            job_id = _batch_submit(
                branch,
                command,
                attempts=1,
                gpu=True,
                parent_job_ids=parent_job_ids)

    def local_run(self, tasks):
        if COMPUTE_RASTER_STATS in tasks:
            run._compute_raster_stats(
                get_config_uri(self.workflow.compute_raster_stats_uri))

        if MAKE_TRAINING_CHIPS in tasks:
            run._make_training_chips(
                get_config_uri(self.workflow.make_training_chips_uri))

        if TRAIN in tasks:
            run._train(get_config_uri(self.workflow.train_uri))

        if PREDICT in tasks:
            run._predict(get_config_uri(self.workflow.predict_uri))

        if EVAL in tasks:
            run._eval(get_config_uri(self.workflow.eval_uri))


def _main(workflow_uri,
          tasks,
          remote=False,
          simulated_remote=False,
          branch='develop',
          run=False):
    if len(tasks) == 0:
        tasks = ALL_TASKS

    for task in tasks:
        if task not in ALL_TASKS:
            raise Exception("Task '{}' is not a valid task.".format(task))

    workflow = ChainWorkflow(workflow_uri, remote=(remote or simulated_remote))
    workflow.save_configs(tasks)

    if run:
        if remote:
            workflow.remote_run(tasks, branch)
        else:
            workflow.local_run(tasks)


@click.command()
@click.argument('workflow_uri')
@click.argument('tasks', nargs=-1)
@click.option('--remote', is_flag=True)
@click.option('--simulated-remote', is_flag=True)
@click.option('--branch', default='develop')
@click.option('--run', is_flag=True)
def main(workflow_uri, tasks, remote, simulated_remote, branch, run):
    _main(
        workflow_uri,
        tasks,
        remote=remote,
        simulated_remote=simulated_remote,
        branch=branch,
        run=run)


if __name__ == '__main__':
    main()
