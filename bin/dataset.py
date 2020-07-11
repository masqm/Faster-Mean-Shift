
import SimpleITK as sitk
from datasets.graph_dataset import GraphDataset
from datasources.cached_image_datasource import CachedImageDataSource
from generators.image_generator import ImageGenerator
from transformations.spatial import translation, scale, composite
from transformations.intensity.sitk.smooth import gaussian
from transformations.intensity.sitk.normalize import normalize_robust as normalize_robust_sitk


class Dataset(object):
    """
    The dataset that processes files from the celltracking challenge.
    """
    def __init__(self,
                 image_size,
                 base_folder='',
                 debug_image_folder=None,
                 data_format='channels_first',
                 image_gaussian_blur_sigma=2.0,
                 image_interpolator='linear',
                 normalization_consideration_factors=(0.2, 0.1),
                 debug_folder_prefix='',
                 additional_scale=None):
        self.image_size = image_size
        self.output_size = self.image_size
        self.base_folder = base_folder
        self.image_base_folder = base_folder
        self.debug_image_folder = debug_image_folder
        self.data_format = data_format
        self.image_gaussian_blur_sigma = image_gaussian_blur_sigma
        self.image_interpolator = image_interpolator
        self.additional_scale = additional_scale
        if self.additional_scale is None:
            self.additional_scale = [1, 1]
        self.normalization_consideration_factors = normalization_consideration_factors
        self.debug_folder_prefix = debug_folder_prefix
        self.dim = 2

    def get_channel_axis(self, image, data_format):
        """
        Returns the channel axis of the given image.
        :param image: The np array.
        :param data_format: The data format. Either 'channels_first' or 'channels_last'.
        :return: The channel axis.
        """
        if len(image.shape) == 3:
            return 0 if data_format == 'channels_first' else 2
        if len(image.shape) == 4:
            return 0 if data_format == 'channels_first' else 3

    def image_postprocessing(self, image):
        """
        Processes input image files. Blurs, crops, pads, and normalizes according to the parameters.
        :param image: The sitk image.
        :return: The processed sitk image.
        """
        if self.image_gaussian_blur_sigma > 0:
            image = gaussian(image, [self.image_gaussian_blur_sigma, self.image_gaussian_blur_sigma])
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        image_float = normalize_robust_sitk(image_float, (-1, 1), consideration_factors=self.normalization_consideration_factors)
        return image_float

    def datasources_single_frame(self, iterator):
        """
        Returns the data sources that load data for a single frame.
        {
        'image:' CachedImageDataSource that loads the image files.
        }
        :return: A dict of data sources.
        """
        datasources_dict = {}
        # image data source loads input image.
        datasources_dict['image'] = CachedImageDataSource(self.image_base_folder, 't', '', '.tif',
                                                          set_identity_spacing=True,
                                                          preprocessing=self.image_postprocessing,
                                                          sitk_pixel_type=sitk.sitkUInt16,
                                                          cache_maxsize=512,
                                                          name='image',
                                                          parents=[iterator])

        return datasources_dict

    def data_generators_single_frame(self, dim, datasources, image_transformation, image_post_processing):
        """
        Returns the data generators that process a single frame. See datasources_single_frame() for dict values.
        :param dim: Image dimension.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :return: A dict of data generators.
        """
        image_size = self.image_size
        data_generators_dict = {}
        data_generators_dict['image'] = ImageGenerator(dim, image_size,
                                                       interpolator=self.image_interpolator,
                                                       post_processing_np=image_post_processing,
                                                       data_format=self.data_format,
                                                       resample_default_pixel_value=-1,
                                                       name='image',
                                                       parents=[datasources['image'], image_transformation])

        return data_generators_dict


    def spatial_transformation(self, image):
        """
        The spatial image transformation without random augmentation.
        :return: The transformation.
        """
        return composite.Composite(self.dim,
                                   [translation.InputCenterToOrigin(self.dim),
                                    scale.Fit(self.dim, self.image_size),
                                    scale.Fixed(self.dim, self.additional_scale),
                                    translation.OriginToOutputCenter(self.dim, self.image_size)],
                                   name='image',
                                   kwparents={'image': image})

    def dataset_val_single_frame(self):
        """
        Returns the validation dataset for single frames. No random augmentation is performed.
        :return: The validation dataset.
        """
        sources = self.datasources_single_frame('id_dict')
        image_key = 'image'
        image_transformation = self.spatial_transformation(sources[image_key])
        generators = self.data_generators_single_frame(2, sources, image_transformation, None)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[image_transformation],
                            iterator='id_dict',
                            debug_image_folder=self.debug_image_folder)
