from setuptools import setup

setup(
    name='odc_index',

    use_scm_version={"root": "../..", "relative_to": __file__},
    setup_requires=['setuptools_scm'],

    author='Open Data Cube',
    author_email='',
    maintainer='Open Data Cube',
    maintainer_email='',

    description='Datacube index prototypes/tools',
    long_description='',

    license='Apache License 2.0',

    tests_require=['pytest'],
    install_requires=[
        'datacube',
        'odc_io',
    ],

    packages=['odc.index'],
    zip_safe=False,
)
