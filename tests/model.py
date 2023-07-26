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
    ...