from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
from pathlib import Path

class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    def extract(self, img):
        # resize the image
        img = img.resize((224, 224))
        img = img.convert('RGB')  # Make sure img is color
        x = image.img_to_array(img)
        # (H, W, C)->(n, H, W, C), where the first elem is the number of img
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize

# if __name__ == '__main__':
#     fe = FeatureExtractor()
#
#     for img_path in sorted(Path("./static/img").glob("*.jpg")):
#         print(img_path)  # e.g., ./static/img/xxx.jpg
#         feature = fe.extract(img=Image.open(img_path))
#         feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
#         np.save(feature_path, feature)