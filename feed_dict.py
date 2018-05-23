import os
import numpy as np
from itertools import cycle

''' 
FeedDict handles several numpy mem_map arrays of image data saved within the directory. The arrays 
should be named in the format "n1_n2.npy" where n1 x n1 is the resolution of the image data in the 
array, and n2 is its number used for indexing purposes. Data should be of type np.float32 and scaled 
between -1.0 and 1.0. In order to avoid loading unnecessary data into memory, only one mem_map is 
loaded at a time.
'''

class FeedDict:
    def __init__(self, directory, shuffle=True):
        files = os.listdir(directory)
        self.arrays = dict()
        for s in [2 ** i for i in range(2, 11)]:
            path_list = []
            for f in files:
                if f.startswith('{}_'.format(s)):
                    path_list.append(os.path.join(directory, f))
            self.arrays.update({s: cycle(path_list)})
            print(path_list)
        self.cur_res = None
        self.cur_path = None
        self.cur_array = None
        self.cur_array_len = 0
        self.idx = 0
        self.shuffle = shuffle

    def _change_res(self, res):
        assert res in self.arrays.keys()
        self.cur_res = res
        self._change_array()

    def _change_array(self):
        new_path = next(self.arrays[self.cur_res])
        if new_path != self.cur_path:
            self.cur_path = new_path
            self.cur_array = np.load(new_path)
            self.cur_array_len = self.cur_array.shape[0]
        if self.shuffle: np.random.shuffle(self.cur_array)
        self.idx = 0

    def next_batch(self, size, res):
        if res != self.cur_res:
            self._change_res(res)
        batch = np.zeros(shape=[size, self.cur_res, self.cur_res, 3], dtype=np.float32)
        for i in range(size):
            batch[i] = self.cur_array[self.idx]
            if self.idx < self.cur_array_len - 1:
                self.idx += 1
            else:
                self._change_array()
        return batch