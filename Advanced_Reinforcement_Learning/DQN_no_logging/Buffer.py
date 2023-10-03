from collections import OrderedDict
import numpy as np
import torch


class BasicBuffer:

    def __init__(self, capacity, fields, wrap=False):
        self.init_buffer(capacity, fields)
        self.clear()

        # Whether the buffer should overwrite old data when full or not.
        self.wrap = wrap

    def clear(self):
        """
        Setting the number of samples in the buffer to 0, 
        and setting the pointer to 0.
        """
        self.ptr = 0
        self.size = 0

    def init_buffer(self, capacity, fields):
        
        # You cannot have an empty list as fields.
        assert fields, '`Fields` must be a non-empty iterable of fields'
        
        # Keeping track of the different fields.
        fields_processed = []
        
        
        for field in fields:
            
            key = field['key']
            
            shape = field.get('shape') or []
            shape = [shape] if np.isscalar(shape) else shape
            
            dtype = field.get('dtype') or None
            dtype = np.float32 if dtype is None else dtype

            fields_processed.append(dict(key=key, shape=shape, dtype=dtype))

        """
        Fields_processed looks like this:
        [{'key': 'ob', 'shape': [4], 'dtype': numpy.float32}, 
        {'key': 'ac', 'shape': [1], 'dtype': numpy.float32},
        {'key': 'rew', 'shape': [], 'dtype': numpy.float32}, 
        {'key': 'next_ob', 'shape': [4], 'dtype': numpy.float32},
        {'key': 'done', 'shape': [], 'dtype': numpy.float32}]
        """

        self.data = {field['key'] : np.zeros(shape=[capacity] + field['shape'], dtype=field['dtype']) for field in fields_processed}
        
        """
        self.data looks like this:
        {
        'ob' : [[0., 0., 0., 0.], [0., 0., 0., 0.], ... _buffer.size_ times],
        'ac' : [[0.], [0.], ... _buffer.size_ times],
        'rew': [0., 0., ... _buffer.size_ times],
        'next_ob': [[0., 0., 0., 0.], [0., 0., 0., 0.], ... _buffer.size_ times],
        'done': [0., 0., ... _buffer.size_ times]
        }
        """
        
    @property
    def keys(self):
        """
        Should return the following:
        ['ob', 'ac', 'rew', 'next_ob', 'done']
        """
        return list(self.data.keys())     

    @property
    def shapes(self):
        """
        Should return the following:
        [(4,), (1,), (), (4,), ()]
        """
        return [v.shape[1:] for v in self.data.values()]

    @property
    def dtypes(self):
        """
        Should return the following:
        [numpy.float32, numpy.float32, numpy.float32, numpy.float32, numpy.float32]
        """
        return [v.dtype for v in self.data.values()]

    @property
    def capacity(self):
        """
        Returns the capacity of the buffer, which is how many samples it can hold.
        The code is a bit tricky, you take the length of the first array in self.data.values().
        """
        return len(next(iter(self.data.values())))

    @property
    def capacity_left(self):
        """
        Tells us how many more samples the buffer can hold.
        """
        return self.capacity - self.size

    @property
    def full(self):
        """
        Tells us whether the buffer is full or not.
        """
        return self.capacity_left <= 0

    def add(self, values):
        
        # Check that values is a dict matching the keys the buffer is set up with
        if not isinstance(values, dict):
            raise ValueError(f'Received non-dict values: {type(values)}')
        
        # Checking that the transition which we want to store in the buffer has the same keys as the buffer.
        if set(self.keys) - set(values):
            raise ValueError(f'Received inconsitent value keys: {set(values)} != {set(self.keys)}')

        # It is a possibility that values has more keys than the buffer.
        # We therefore only keep the keys which the buffer has.
        values = {k: values[k] for k in self.keys}

        # Check that all values have the same length
        # There are cases where you want to store more than one transition in the buffer at the same time.
        lengths = [len(v) for v in values.values()]
        length = lengths[0]
        

        if length == 0:
            # No information to add, you could raise an error here instead.
            return

        # All the values should have same length.
        if not all(l == length for l in lengths):
            raise ValueError(f'Received inconsistent value lengths: {lengths}')

        # If the buffer can fit the data without looping the pointer, fill it in
        if self.ptr + length <= self.capacity:

            # For each key, value pair (example: 'ob', [[1, 1, 1, 1]])
            for key, value in values.items():
                # Add the data to the buffer.
                self.data[key][self.ptr:self.ptr + length] = value
            
            # Update the size and pointer, based on how many samples we added.
            self.size = min(self.size + length, self.capacity); self.ptr = self.ptr + length

            # In the special case where the buffer is full, we set the pointer to 0.
            self.ptr = self.ptr % self.capacity if self.wrap else self.ptr

        # In the case where the buffer capacity will be exceeded, we can either wrap or raise an error.
        elif self.wrap:
            
            i = 0
            while i < length:
                """
                This is a bit advanced.
                We essentially split the data up into two chunks, and add them separately.
                j represents the maximum index we can add to the buffer without exceeding the capacity.
                In the second iteration of the while-loop, we add the remaining data.
                """
                j = i + min(length - i, self.capacity - self.ptr)
                values_chunk = {key: value[i:j] for key, value in values.items()}
                self.add(values_chunk)
                i = j

        # If we have exceeded the capacity with wrap=False, raise an error
        else:
            raise ValueError('Buffer capacity exceeded with wrap=False')

    def get(self, indices=None, as_dict=False, to_torch=True, device='cpu'):
        """
        Indices: The indices of the samples we want to get from the buffer.
        as_dict: Whether we want the data as a dictionary or a tuple.
        to_torch: Whether we want to convert the data to torch tensors.
        device: The device we want to store the data on, either cpu or gpu.
        """

        # If indices is None, we want to get all the samples in the buffer.
        indices = slice(0, len(self)) if indices is None else indices

        # Extracting data from the buffer, based on the indices.
        # You will get something like this:
        # {'ob': [[1, 1, 1, 1], [2, 2, 2, 2]], 'ac': [1, 2], 'rew': [1, 1], 'next_ob': [[2, 2, 2, 2], [3, 3, 3, 3]], 'done': [0, 0]}
        data = {k: v[indices] for k, v in self.data.items()}
        # If we are converting to torch tensors, we do that here.
        if to_torch:
            data = {k: torch.from_numpy(v).to(device) for k, v in data.items()}        
        
        # If we want the data as a dictionary, we simply return the data dict.
        # If we want the data as a tuple, we return a tuple of the data values.
        # An example of the latter would be:
        # (array([[1, 1, 1, 1], [2, 2, 2, 2]]), array([1, 2]), array([1, 1]), array([[2, 2, 2, 2], [3, 3, 3, 3]]), array([0, 0]))
        return data if as_dict else tuple(data[k] for k in self.keys)

    def sample(self, batch_size=1, replace=True, **kwargs):
        indices = np.random.choice(self.size, size=batch_size, replace=replace)
        return self.get(indices=indices, **kwargs)


    # Have not looked into these two methods yet.
    def sample_recent(self, n, batch_size=1, replace=True, **kwargs):
        indices = np.random.choice(self.size - n, size=batch_size, replace=replace)
        return self.get(indices=n + indices, **kwargs)

    def pop(self, **kwargs):
        data = self.get(**kwargs)
        self.clear()
        return data

    @classmethod
    def make_default(cls, capacity, ob_dim, ac_dim, **kwargs):
        """
        Creates a buffer with default fields for observations, action, reward and done.
        Keywordargument is used in our case to set wrap=True.
        """
        return cls(
            capacity = capacity,
            fields=[
                dict(key='ob', shape=ob_dim),
                dict(key='ac', shape=ac_dim),
                dict(key='rew'),
                dict(key='next_ob', shape=ob_dim),
                dict(key='done'),
            ],
            **kwargs
        )

    def __len__(self):
        """
        Returns the number of samples currently in the buffer.
        """
        return self.size

    def __getitem__(self, key):
        """
        Returns the stored data for the given key.
        """
        return self.data[key][:self.size]

    def __lshift__(self, values):
        """
        Makes us able to store a transition in the buffer with this syntax: buffer << values.
        """
        return self.add(values)

    def __repr__(self):
        """
        String representation of the buffer.
        """
        return f'<BasicBuffer {self.size}/{self.capacity}>'


if __name__ == '__main__':
    buffer = BasicBuffer(10, fields=[dict(key='ob', shape=[4]), dict(key='ac', shape=1)])

    print(buffer)

    buffer << {'ob': [[1, 1, 1, 1]], 'ac': [1]}
    buffer << {'ob': [[2, 2, 2, 2]], 'ac': [2]}

    print(buffer)
    print(buffer.sample(1))
