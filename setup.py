# Copyright (c) 2020 Jeff Irion and contributors

"""Setup information for the ``graphslam`` package.

"""


from setuptools import setup

with open('README.rst') as f:
    readme = f.read().replace(':math:', '').replace('\\mathbb{R}', 'R')

setup(
    name='graphslam',
    version='0.0.3',
    description='Graph SLAM solver in Python',
    long_description=readme,
    keywords=['graphslam', 'slam', 'graph', 'optimization', 'mapping'],
    url='https://github.com/JeffLIrion/python-graphslam',
    license='MIT',
    author='Jeff Irion',
    author_email='jefflirion@users.noreply.github.com',
    packages=['graphslam', 'graphslam.pose', 'graphslam.edge'],
    install_requires=['numpy', 'scipy'],
    # tests_require=[],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3'],
    test_suite='tests'
)
