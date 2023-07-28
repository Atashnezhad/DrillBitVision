class ImageObject:
    def ImageDataGenerator(self, *args, **kwargs):
        # print(args, kwargs)
        return self

    def flow(self, *args, **kwargs):
        # print(args, kwargs)
        return []


class XObjClass:
    @staticmethod
    def shape():
        return (1, 2, 3)

    def reshape(self, *args, **kwargs):
        # print(args, kwargs)
        return self


class ImageAddressObject:
    @property
    def name(self):
        return "test_image.jpg"


class TestAugmentData2Mock:
    def categorie_name(self, *args, **kwargs):
        return ["pdc_bit", "rollercone_bit"]

    @staticmethod
    def image_dict(*args, **kwargs):
        return {
            "pdc_bit": {"image_list": [ImageAddressObject], "number_of_images": 0},
            "rollercone_bit": {
                "image_list": [ImageAddressObject],
                "number_of_images": 0,
            },
        }

    @staticmethod
    def load_img(*args, **kwargs):
        return None

    def flow(self, *args, **kwargs):
        return []

    @staticmethod
    def img_to_array_func(*args, **kwargs):
        # print(args, kwargs)
        return XObjClass
