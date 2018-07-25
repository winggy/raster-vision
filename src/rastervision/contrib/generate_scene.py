import random

import click
import numpy as np
import rasterio
from rasterio.transform import from_origin

from rastervision.core.box import Box
from rastervision.crs_transformers.rasterio_crs_transformer import (
    RasterioCRSTransformer)
from rastervision.labels.object_detection_labels import ObjectDetectionLabels
from rastervision.label_stores.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile)
from rastervision.core.class_map import ClassItem, ClassMap


@click.command()
@click.argument('tiff_path')
@click.argument('labels_path')
def generate_scene(tiff_path, labels_path):
    # make extent that's divisible by chip_size
    chip_size = 300
    ymax = 3 * chip_size
    xmax = 3 * chip_size
    extent = Box(0, 0, ymax, xmax)

    # make windows along grid
    windows = extent.get_windows(chip_size, chip_size)

    # for each window, make some random boxes within it and render to image
    nb_channels = 3
    image = np.zeros((ymax, xmax, nb_channels)).astype(np.uint8)
    boxes = []
    class_ids = []
    for window in windows:
        # flip coin to see if window has box in it
        if random.uniform(0, 1) > 0.5:
            box = window.make_random_square(50).as_int()
            class_id = random.randint(1, 2)

            boxes.append(box)
            class_ids.append(class_id)

            image[box.ymin:box.ymax, box.xmin:box.xmax, class_id - 1] = 255

    # save image as geotiff centered in philly
    transform = from_origin(-75.163506, 39.952536, 0.000001, 0.000001)

    with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=ymax,
            transform=transform,
            crs='EPSG:4326',
            compression=rasterio.enums.Compression.none,
            width=xmax,
            count=nb_channels,
            dtype='uint8') as dst:
        for channel_ind in range(0, nb_channels):
            dst.write(image[:, :, channel_ind], channel_ind + 1)

    # make an OD labels and make boxes
    npboxes = Box.to_npboxes(boxes)
    class_ids = np.array(class_ids)
    labels = ObjectDetectionLabels(npboxes, class_ids)

    # save labels to geojson
    image_dataset = rasterio.open(tiff_path)
    crs_transformer = RasterioCRSTransformer(image_dataset)
    class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'building')])
    od_file = ObjectDetectionGeoJSONFile(
        labels_path, crs_transformer, class_map, readable=False, writable=True)
    od_file.set_labels(labels)
    od_file.save()


if __name__ == '__main__':
    generate_scene()
