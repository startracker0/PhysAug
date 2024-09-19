# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class DiverseWeatherDataset(XMLDataset):
    """Dataset for DiverseWeatherDataset."""

    METAINFO = {
        'classes':
        ('bus', 'bike', 'car', 'motor', 'person', 'rider', 'truck'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142)]
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._metainfo['dataset_type'] = 'DiverseWeatherDataset'
      