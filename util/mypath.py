import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10'}
        assert(database in db_names)

        if database == 'cifar-10':
            return 'dataset/cifar10'

        else:
            raise NotImplementedError
