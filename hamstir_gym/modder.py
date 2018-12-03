import pybullet as p
import numpy as np

from hamstir_gym.utils import DATA_DIR

class Modder:
    def __init__(self, h=256, w=256):
        self.h,self.w = h, w
        self.pixels = np.zeros((h,w,3),dtype=np.int32)
        
    def load(self, parent):
        self.parent = parent
        num_planes = p.getNumJoints(parent)
        self.joints = [-1] + list(range(num_planes))
        self.textures = []
        for j in self.joints:
            p.changeVisualShape(parent,j,rgbaColor=[1,1,1,1])
            self.textures.append(p.loadTexture(DATA_DIR+"tex256.png"))
            p.changeVisualShape(parent,j,textureUniqueId=self.textures[-1])
    
    def randomRGB(self):
        return np.random.uniform(size=3)*256
    
    def randomize(self):
        for t in self.textures:
            p.changeTexture(t,self.random_pixels(),self.w,self.h)
            
    def random_pixels(self):
        choices = [
            self.rand_checker,
            self.rand_gradient,
            self.rand_uniform,
            self.rand_noise,
        ]
        choice = np.random.randint(len(choices))
        return choices[choice]()
        
    def rand_checker(self):
        checker_size = 2 ** np.random.randint(3,7)
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        for i in range(self.h):
            for j in range(self.w):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    self.pixels[i][j] = rgb1
                else:
                    self.pixels[i][j] = rgb2
        return self.pixels.flatten().tolist()
        
    def rand_gradient(self):
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        vertical = np.random.randint(2)
        if vertical == 1:
            for j in range(self.w):
                frac = float(j)/(self.w-1.0)
                self.pixels[:][j] = rgb1*(1-frac) + frac*rgb2
        else:
            for i in range(self.h):
                frac = float(i)/(self.h-1.0)
                self.pixels[i][:] = rgb1*(1-frac) + frac*rgb2
        return self.pixels.flatten().tolist()
        
    def rand_uniform(self):
        rgb = self.randomRGB()
        self.pixels[:][:] = rgb
        return self.pixels.flatten().tolist()
        
    def rand_noise(self):
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        fraction = 0.1 + np.random.uniform() * 0.8
        mask = np.random.uniform(size=(self.h,self.w)) > fraction
        self.pixels[..., :] = rgb1
        self.pixels[mask, :] = rgb2
        return self.pixels.flatten().tolist()
        