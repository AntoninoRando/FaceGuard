import numpy as np

def compute_lbp(image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute Local Binary Pattern for texture analysis"""
        h, w = image.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_string = ''
                
                # 8 neighbors in circular pattern
                neighbors = [
                    (i-radius, j-radius), (i-radius, j), (i-radius, j+radius),
                    (i, j+radius), (i+radius, j+radius), (i+radius, j),
                    (i+radius, j-radius), (i, j-radius)
                ]
                
                for ni, nj in neighbors:
                    binary_string += '1' if image[ni, nj] >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp