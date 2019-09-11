import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    config = Configuration('DMRecall', parent_package, top_path)

    # submodules with build utilities
    config.add_subpackage('algorithm')
    config.add_subpackage('baseclass')
    config.add_subpackage('evaluation')
    config.add_subpackage('dataset')
    config.add_subpackage('tools')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())