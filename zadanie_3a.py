import os
import cv2
from numpy import *
from pandas import *
from skimage.feature import graycomatrix, graycoprops
class Processor:
    def __init__(self, input: str, output: str, crop_size: float):
        self._in: str = input
        self._out: str = output
        self._crops: float = crop_size
    def get_category(self, file) -> tuple[str, str]:
        name = os.path.splitext(file)[0]
        folder = os.path.join(self._out, name)
        return name, folder
    def crop_textures(self):
        # Lista przechowująca wektory cech dla wszystkich obrazów
        vfeatures = []
        features_name = lambda category, i, j: f"{category}.{i}.{j}.data.txt"
        makedir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
        filenames = [(cv2.imread(os.path.join(self._in, file)),file) for file in os.listdir(self._in) if any(file.endswith(f"*.{extension}") for extension in "jpg png".split())]
        min_size = min(min([min(image.shape[:2]) for image, _ in filenames]), float('inf'))
        num_x = num_y = (min_size // self._crops)
        cat_name, cat_dir = self.get_category()

        """
            1. crop images into smaller fragments
            2. greyscale 'em
        """
        for image, file in filenames:
            image = cv2.imread(os.path.join(self._in, file))
            makedir(cat_dir)

            for i in range(0, num_y * self._crops, self._crops):
                for j in range(0, num_x * self._crops, self._crops):
                    crop = image[i:i+self._crops, j:j+self._crops]
                    crop_dir = os.path.join(cat_dir, "crop")
                    makedir(crop_dir)
                    cv2.imwrite(os.path.join(crop_dir, f"{cat_name}_{i}.{j}.part.jpg"), crop)
    

                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    gray_dir = os.path.join(cat_dir, "grayscale")
                    if not os.path.exists(gray_dir):
                        os.makedirs(gray_dir)
                    cv2.imwrite(os.path.join(gray_dir, f"{cat_name}_{i}.{j}.grayscale.jpg"), gray_crop)
                    glcm = graycomatrix(gray_crop, distances=[1, 3, 5], angles=[0, pi/4, pi/2, 3*pi/4], symmetric=True, normed=True)
    
                    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
                    correlation = graycoprops(glcm, 'correlation').ravel()
                    contrast = graycoprops(glcm, 'contrast').ravel()
                    energy = graycoprops(glcm, 'energy').ravel()
                    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
                    asm = graycoprops(glcm, 'ASM').ravel()

                    with open(os.path.join(cat_dir, features_name(cat_name, i, j)), 'w') as f:
                        f.write(
                            f"Dissimilarity: {dissimilarity}\n"
                            f"Correlation: {correlation}\n"
                            f"Contrast: {contrast}\n"
                            f"Energy: {energy}\n"
                            f"Homogeneity: {homogeneity}\n"
                            f"ASM: {asm}\n"
                        )
                    vfeatures.append({
                        "Category": cat_name,
                        "File": features_name(cat_name, i, j),
                        "Dissimilarity": dissimilarity,
                        "Correlation": correlation,
                        "Contrast": contrast,
                        "Energy": energy,
                        "Homogeneity": homogeneity,
                        "ASM": asm
                    })

        df = DataFrame(vfeatures)
        csv_file_path = os.path.join(self._out, 'feature_vectors.csv')
        df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    from shutil import rmtree
    rmtree("data\out")
    os.makedirs(r"data\out")
    proc = Processor(r"data\in", r"data\out", 128)
    proc.crop_textures()