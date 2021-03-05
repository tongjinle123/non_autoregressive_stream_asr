import numpy as np
import h5py as hpy
from numpy.core.fromnumeric import shape
from tqdm import tqdm 


vlen_float32 = hpy.special_dtype(vlen=np.float32)
vlen_int32 = hpy.special_dtype(vlen=np.int32)

class Builder:
    def __init__(self):
        self.dtype = np.dtype(
            [
                ('feature', vlen_float32),
                ('feature_length', np.int32),
                ('feature_size', np.int32),
                ('target', vlen_int32),
            ]
        )

    def build(self, file_name, line_iter, dataset_name='dataset', ):
        chunk_num = 1
        with hpy.File(file_name, 'w') as writer:
            dataset = writer.create_dataset(dataset_name, shape=(1, ), dtype=self.dtype, maxshape=(None,), compression='gzip')
            for index, data in enumerate(line_iter):
                if data != None:
                    dataset[chunk_num-1] = data
                    writer.flush()
                    chunk_num += 1
                    dataset.resize(chunk_num, 0)
            dataset.resize(chunk_num - 1, 0)



if __name__ == '__main__':
    
    def line_iter():
        for i in range(1000):
            length = np.random.randint(100, 200)
            size = 80
            target_length = np.random.randint(20,50)
            feature = np.random.randn(length, size)
            target = np.zeros(target_length, dtype=np.int32)
            yield feature.reshape(-1), length, size, target
    
    builder = Builder()
    builder.build('test.h', line_iter(), 'datas')