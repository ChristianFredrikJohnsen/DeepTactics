class Config:

    def __init__(self, **kwargs):
        for k, v in self.get_members().items():
            setattr(self, k, kwargs.pop(k, v))
        if kwargs:
            raise ValueError(f'Recieved unexpected arguments: {kwargs}')

    @classmethod
    def get_members(cls):
        return {
            k: v for cls in cls.mro()[:-1][::-1]
            for k, v in vars(cls).items()
            if not callable(v) and
                not isinstance(v, (property, classmethod, staticmethod))
                and not k.startswith('__')
        }


    @property
    def dict(self):
        return{k: getattr(self, k) for k in self.get_members()}



