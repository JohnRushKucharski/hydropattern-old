'''Required to build hydropattern as a package.'''

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hydropattern',
    version='0.0.0',
    description='Finds natural flow regime type patterns in time series data .',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/JohnRushKucharski/hydropattern',
    author='JohnRushKucharski',
    author_email='johnkucharski@gmail.com',
    license='GNU GPLv3',
    classifiers=['License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Programming Language :: Python :: 3.12',
                 'Operating System :: OS Independent'],
    install_requires=['numpy'
                      'pandas'
                      ],
    extras_require={'dev': ['twine']
                    },
    python_requires='>=3.12',
)