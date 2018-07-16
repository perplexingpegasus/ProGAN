import os
import pickle
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

    pickle_filename = 'fd_log.plk'

    def __init__(self, imgdir, logdir, shuffle=True, min_size=4, max_size=1024):

        self.logdir = logdir
        self.shuffle = shuffle
        self.sizes = [2 ** i for i in range(
            int(np.log2(min_size)),
            int(np.log2(max_size)) + 1
        )]

        files = os.listdir(imgdir)
        self.arrays = dict()

        for s in [2 ** i for i in range(2, 11)]:
            path_list = []
            for f in files:

                if f.startswith('{}_'.format(s)):
                    path_list.append(os.path.join(imgdir, f))

            if shuffle: np.random.shuffle(path_list)
            self.arrays.update({s: cycle(path_list)})

        self.cur_res = None
        self.cur_path = None
        self.cur_array = None
        self.cur_array_len = 0
        self.idx = 0

    @property
    def n_sizes(self): return len(self.sizes)

    def __change_res(self, res):
        assert res in self.arrays.keys()
        self.cur_res = res
        self.__change_array()

    def __change_array(self):
        new_path = next(self.arrays[self.cur_res])
        print('Loaded new memmap array: {}'.format(new_path))
        if new_path != self.cur_path:
            self.cur_path = new_path
            self.cur_array = np.load(new_path)
            self.cur_array_len = self.cur_array.shape[0]
        if self.shuffle: np.random.shuffle(self.cur_array)
        self.idx = 0

    def next_batch(self, batch_size, res):
        if res != self.cur_res:
            self.__change_res(res)

        remaining = self.cur_array_len - self.idx
        start = self.idx

        if remaining >= batch_size:
            stop = start + batch_size
            batch = self.cur_array[start:stop]

        else:
            stop = batch_size - remaining
            batch = self.cur_array[start:]
            self.__change_array()
            batch = np.concatenate((batch, self.cur_array[:stop]))

        self.idx = stop

        return batch

    @classmethod
    def load(cls, imgdir, logdir):
        path = os.path.join(logdir, cls.pickle_filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                fd = pickle.load(f)
            if type(fd) == cls:
                return fd
        return cls(imgdir, logdir)

    def save(self):
        path = os.path.join(self.logdir, self.pickle_filename)
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)