import setuptools

NUMPY_MIN_VERSION = '1.16.0'
SCIPY_MIN_VERSION = '1.2.0'
JOBLIB_MIN_VERSION = '0.11a1'
GENSIM_MIN_VERSION = '3.8.0'
NETWORKX_MIN_VERSION = '2.2'
PANDAS_MIN_VERSION = '0.24.0'

with open("README.md", "r") as fh:
    long_description = fh.read()


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.add_subpackage('DMRecall')
    return config


def setup_package():
    metadata = dict(
        name="DMRecall",
        version="1.1.1",
        author="Wen Yi",
        author_email="wenyi@cvte.com",
        description="Recommendation algorithm",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://www.cvte.com",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        install_requires = ['numpy>={}'.format(NUMPY_MIN_VERSION),
                            'scipy>={}'.format(SCIPY_MIN_VERSION),
                            'joblib>={}'.format(JOBLIB_MIN_VERSION),
                            'gensim>={}'.format(GENSIM_MIN_VERSION),
                            'networkx>={}'.format(NETWORKX_MIN_VERSION),
                            'pandas>={}'.format(PANDAS_MIN_VERSION)
                           ]
    )
    metadata['configuration'] = configuration

    setuptools.setup(**metadata)


if __name__ == "__main__":
    setup_package()