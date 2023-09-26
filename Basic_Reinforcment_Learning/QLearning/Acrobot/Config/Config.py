class Config:

    def __init__(self, **kwargs):
        """
        Initializes all the class attributes.
        If a keyword argument is used, it will override the default value.
        """
        
        for k, v in self.get_members().items():
            """
            Similar to self.k = kwargs.pop(k, v)
            if k in kwargs:
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, v)
            """
            setattr(self, k, kwargs.pop(k, v))
            
        if kwargs:
            raise ValueError(f'Received unexpected arguments: {kwargs}')

    @classmethod
    def get_members(cls):
        """Returns a dictionary containing all class attributes."""
        return {
            k: v for cls in cls.mro()[:-1][::-1]
            for k, v in vars(cls).items()
            if not callable(v) and
                not isinstance(v, (property, classmethod, staticmethod))
                and not k.startswith('__')
        }


    @property
    def dict(self):
        """Returns a dictionary containing all class attributes."""
        return{k: getattr(self, k) for k in self.get_members()}

print(dir(Config))