def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('dataset', parent_package, top_path)

    # submodules with build utilities
    config.add_data_dir('sample_data')
    config.add_subpackage('load_data')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())