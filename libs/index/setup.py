from setuptools import setup

setup(
    name='odc_index',

    version='1',
    author='Open Data Cube',
    author_email='',
    maintainer='Open Data Cube',
    maintainer_email='',

    description='Various Parallel Processing Tools',
    long_description='',

    license='Apache License 2.0',

    tests_require=['pytest'],
    install_requires=[
        'datacube',
        'odc_io @ git+https://github.com/opendatacube/dea-proto.git#egg=odc_io&subdirectory=libs/io',
    ],

    packages=['odc.index'],
    zip_safe=False,
)
