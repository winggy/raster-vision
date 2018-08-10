from os.path import join

from rastervision.workflows.config_utils import (
    make_model_config, CL, OD, ClassificationGeoJSONOptions)
from rastervision.workflows.chain_utils import (
    ChainWorkflowPaths, ChainWorkflowConfigGenerator,
    ChainWorkflow)
from rastervision.workflows.backend_utils import (
    get_pretrained_model_uri, save_backend_config, make_backend_config_uri)


class CowcPotsdamPaths():
    def __init__(self, isprs_potsdam_uri, cowc_potsdam_uri):
        self.isprs_potsdam_uri = isprs_potsdam_uri
        self.cowc_potsdam_uri = cowc_potsdam_uri

    def make_image_uri(self, id):
        return join(
            self.isprs_potsdam_uri,
            '4_Ortho_RGBIR/top_potsdam_{id}_RGBIR.tif'.format(id=id))

    def make_label_uri(self, id):
        return join(
            self.cowc_potsdam_uri,
            'labels/all/top_potsdam_{id}_RGBIR.json'.format(id=id))


class TinyCowcPotsdamPaths():
    def __init__(self, cowc_potsdam_uri):
        self.cowc_potsdam_uri = cowc_potsdam_uri

    def make_image_uri(self, id):
        return join(
            self.cowc_potsdam_uri,
            'tiny/{id}.tif'.format(id=id))

    def make_label_uri(self, id):
        return join(
            self.cowc_potsdam_uri,
            'tiny/{id}.json'.format(id=id))


def generate_workflow(workflow_uri, isprs_potsdam_uri, cowc_potsdam_uri,
                      use_tiny_dataset=False, task=OD):
    class_names = ['car']
    # if task == CL:
    #    class_names.append('background')
    train_scene_ids = [
        '2_10', '2_11', '2_12', '2_14', '3_11', '3_13', '4_10', '5_10', '6_7',
        '6_9']
    validation_scene_ids = ['2_13', '6_8', '3_10']
    cowc_potsdam_paths = CowcPotsdamPaths(isprs_potsdam_uri, cowc_potsdam_uri)

    if use_tiny_dataset:
        train_scene_ids = train_scene_ids[0:1]
        validation_scene_ids = validation_scene_ids[0:1]
        cowc_potsdam_paths = TinyCowcPotsdamPaths(cowc_potsdam_uri)

    paths = ChainWorkflowPaths(workflow_uri)
    model_config = make_model_config(class_names, task)
    config_generator = ChainWorkflowConfigGenerator(paths)

    def make_scene(id):
        # task_options = ClassificationGeoJSONOptions(background_class_id=2)
        return config_generator.make_geotiff_geojson_scene(
            id, [cowc_potsdam_paths.make_image_uri(id)], task,
            ground_truth_labels_uri=cowc_potsdam_paths.make_label_uri(id))

    train_scenes = [make_scene(id) for id in train_scene_ids]
    validation_scenes = [make_scene(id) for id in validation_scene_ids]

    workflow = ChainWorkflow(
        paths, model_config, train_scenes, validation_scenes)
    workflow.save()


def main():
    # Set these fields
    workflow_uri = '/opt/data/lf-dev/cowc-potsdam'
    isprs_potsdam_uri = '/opt/data/raw-data/isprs-potsdam'
    cowc_potsdam_uri = '/opt/data/lf-dev/processed-data/cowc-potsdam'
    use_tiny_dataset = True
    task = OD

    generate_workflow(
        workflow_uri, isprs_potsdam_uri, cowc_potsdam_uri,
        use_tiny_dataset, task)


if __name__ == '__main__':
    main()
