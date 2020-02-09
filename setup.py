"""Setup information for the ``graphslam`` package.

"""


from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='graphslam',
    version='0.0.1',
    description='Graph SLAM solver in Python',
    long_description=readme,
    keywords=['graphslam', 'slam'],
    url='https://github.com/JeffLIrion/python-graphslam',
    license='MIT',
    author='Jeff Irion',
    author_email='jefflirion@users.noreply.github.com',
    packages=['graphslam', 'graphslam.pose'],
    install_requires=['numpy', 'scipy'],
    # tests_require=[],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3'],
    test_suite='tests'
)
