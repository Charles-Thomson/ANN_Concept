def resettable(f):
    import copy

    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)
        self.__origional_dict__ = copy(self.__dict__)

        def reset(o=self):
            o.__dict__ = o.__origional_dict__

        self.reset = reset

    return __init_and_copy__
