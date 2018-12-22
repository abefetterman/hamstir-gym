import pybullet as p
import numpy as np
from gym.utils import seeding

from hamstir_gym.utils import DATA_DIR

class Modder:
    def __init__(self, h=256, w=256):
        self.h,self.w = h, w
        self.np_pixels = np.zeros((h,w,3),dtype=np.int32)
        self.pixels = [0]*self.h*self.w*3
        self.seed()
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed
        
    def load(self, parent):
        self.parent = parent
        num_planes = p.getNumJoints(parent)
        self.joints = [-1] + list(range(num_planes))
        self.textures = []
        for j in self.joints:
            p.changeVisualShape(parent,j,rgbaColor=[1,1,1,1])
            t = p.loadTexture(DATA_DIR+"tex256.png")
            self.textures.append(t)
            p.changeTexture(t,self.random_pixels(),self.w,self.h)
            p.changeVisualShape(parent,j,textureUniqueId=t)
            
    def hide(self):
        for j in self.joints:
            p.changeVisualShape(self.parent,j,rgbaColor=[1,1,1,0])
            
    def show(self):
        for j in self.joints:
            p.changeVisualShape(self.parent,j,rgbaColor=[1,1,1,1])
    
    def randomRGB(self):
        return self.np_random.uniform(size=3)*256
    
    def randomize(self):
        for t in self.textures:
            p.changeTexture(t,self.random_pixels(),self.w,self.h)
    
    def copy_np_pixels(self):
        f = self.np_pixels.flatten()
        for i in range(self.h*self.w*3):
            self.pixels[i] = f[i]
        return self.pixels
            
    def random_pixels(self):
        choices = [
            self.rand_checker,
            self.rand_gradient,
            self.rand_uniform,
            self.rand_noise,
        ]
        choice = self.np_random.randint(len(choices))
        return choices[choice]()
        
    def rand_checker(self):
        checker_size = 2 ** self.np_random.randint(3,7)
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        for i in range(self.h):
            for j in range(self.w):
                if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                    self.np_pixels[i][j] = rgb1
                else:
                    self.np_pixels[i][j] = rgb2
        return self.copy_np_pixels()
        
    def rand_gradient(self):
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        vertical = self.np_random.randint(2)
        if vertical == 1:
            for j in range(self.w):
                frac = float(j)/(self.w-1.0)
                self.np_pixels[:][j] = rgb1*(1-frac) + frac*rgb2
        else:
            for i in range(self.h):
                frac = float(i)/(self.h-1.0)
                self.np_pixels[i][:] = rgb1*(1-frac) + frac*rgb2
        return self.copy_np_pixels()
        
    def rand_uniform(self):
        rgb = self.randomRGB()
        self.np_pixels[:][:] = rgb
        return self.copy_np_pixels()
        
    def rand_noise(self):
        rgb1, rgb2 = self.randomRGB(), self.randomRGB()
        fraction = 0.1 + self.np_random.uniform() * 0.8
        mask = self.np_random.uniform(size=(self.h,self.w)) > fraction
        self.np_pixels[..., :] = rgb1
        self.np_pixels[mask, :] = rgb2
        return self.copy_np_pixels()
        