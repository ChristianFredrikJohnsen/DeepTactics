import collections

import numpy as np
import torch


class BasicBuffer:

    def __init__(self, capacity, fields, wrap=False):
        self.init_buffer(capacity, fields)
        self.clear()
        self.wrap = wrap

    def clear(self):
        self.ptr = 0
        self.size = 0

    def init_buffer(self, capacity, fields):
        assert fields, '`Fields` must be a non-empty iterable of fields'
        fields_processed = []
        for field in fields:
            key = field['key']
            shape = field.get('shape') or []
            shape = [shape] if np.isscalar(shape) else shape
            dtype = field.get('dtype') or None
            dtype = np.float32 if dtype is None else dtype

            fields_processed.append(dict(key=key, shape=shape, dtype=dtype))

        self.data = collections.OrderedDict(
            (field['key'],
             np.zeros(
                 shape=[capacity] + field['shape'], dtype=field['dtype']
             )) for field in fields_processed
        )

    @property
    def keys(self):
        return list(self.data.keys())

    @property
    def shapes(self):
        return [v.shape[1:] for v in self.data.values()]

    @property
    def dtypes(self):
        return [v.dtype for v in self.data.values()]

    @property
    def capacity(self):
        return len(next(iter(self.data.values())))

    @property
    def capacity_left(self):
        return self.capacity - self.size

    @property
    def full(self):
        return self.capacity_left <= 0

    def add(self, values):
        # Check that values is a dict matching the keys the buffer is set up with
        if not isinstance(values, dict):
            raise ValueError(f'Received non-dict values: {type(values)}')
        if set(self.keys) - set(values):
            raise ValueError(
                f'Received inconsitent value keys: '
                f'{set(values)} != {set(self.keys)}'
            )

        values = {k: values[k] for k in self.keys}
        # Check that all values have the same length
        lengths = [len(v) for v in values.values()]
        length = lengths[0]
        if length == 0:
            return

        if not all(l == length for l in lengths):
            raise ValueError(f'Received inconsistent value lengths: {lengths}')

        # If the buffer can fit the data without looping the pointer, fill it in
        if self.ptr + length <= self.capacity:
            for key, value in values.items():
                self.data[key][self.ptr:self.ptr + length] = value
            self.size = min(self.size + length, self.capacity)
            self.ptr = self.ptr + length

            if self.wrap:
                self.ptr = self.ptr % self.capacity

        elif self.wrap:
            i = 0
            while i < length:
                j = i + min(length - i, self.capacity - self.ptr)
                values_chunk = {
                    key: value[i:j] for key, value in values.items()
                }
                self.add(values_chunk)
                i = j

        # Otherwise, complain
        else:
            raise ValueError('Buffer capacity exceeded with wrap=False')

    def get(self, indices=None, as_dict=False, to_torch=True, device=None):
        indices = slice(0, len(self)) if indices is None else indices
        data = {k: v[indices] for k, v in self.data.items()}
        if to_torch:
            data = {
                k: torch.from_numpy(v).to(device) for k, v in data.items()
            }
        if as_dict:
            return data
        else:
            return tuple(data[k] for k in self.keys)

    def sample(self, batch_size=1, replace=True, **kwargs):
        indices = np.random.choice(self.size, size=batch_size, replace=replace)
        return self.get(indices=indices, **kwargs)

    def sample_recent(self, n, batch_size=1, replace=True, **kwargs):
        indices = np.random.choice(self.size - n, size=batch_size, replace=replace)
        return self.get(indices=n + indices, **kwargs)

    def pop(self, **kwargs):
        data = self.get(**kwargs)
        self.clear()
        return data

    @classmethod
    def make_default(cls, capacity, ob_dim, ac_dim, **kwargs):
        return cls(
            capacity=capacity,
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
        return self.size

    def __getitem__(self, k):
        return self.data[k][:self.size]

    def __lshift__(self, values):
        return self.add(values)

    def __repr__(self):
        return f'<BasicBuffer {self.size}/{self.capacity}>'


if __name__ == '__main__':
    buffer = BasicBuffer(10, fields=[dict(key='ob', shape=[4]), dict(key='ac', shape=1)])

    print(buffer)

    buffer << {'ob': [[1, 1, 1, 1]], 'ac': [1]}
    buffer << {'ob': [[2, 2, 2, 2]], 'ac': [2]}

    print(buffer)
    print(buffer.sample(1))
