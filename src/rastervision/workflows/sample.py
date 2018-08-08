from rastervision.workflows.config_utils import (
    make_model_config, CL)
from rastervision.workflows.chain_utils import (
    ChainWorkflowPaths, ChainWorkflowSceneGenerator, ChainWorkflow)


def main():
    base_uri = '/opt/data/lf-dev/workflow-dreams/'
    paths = ChainWorkflowPaths(base_uri)
    chip_size = 300
    debug = True
    model_config = make_model_config(
        ['car', 'building'], CL)
    scene_generator = ChainWorkflowSceneGenerator(paths, chip_size=chip_size)

    def make_scene(id, raster_uris, ground_truth_labels_uri=None):
        return scene_generator.make_classification_geotiff_geojson_scene(
            id,
            raster_uris,
            ground_truth_labels_uri=ground_truth_labels_uri,
            channel_order=[2, 1, 0])

    train_scenes = [
        make_scene('2-10', ['/test/2-10.tif'], '/test/2-10.json'),
        make_scene('2-11', ['/test/2-11.tif'], '/test/2-11.json')
    ]
    validation_scenes = [
        make_scene('2-12', ['/test/2-12.tif'], '/test/2-12.json')
    ]

    workflow = ChainWorkflow(
        paths, model_config, train_scenes, validation_scenes,
        chip_size=chip_size, debug=debug)
    workflow.save_config()


if __name__ == '__main__':
    main()
