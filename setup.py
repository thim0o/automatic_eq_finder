# setup.py

from setuptools import setup, find_packages

setup(
    name='automatic-eq-finder',
    version='1.0.0',
    url='https://github.com/thim0o/automatic_eq_finder',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'pyaudio',
        'matplotlib',
        'PyQt5',
    ],
    entry_points={
        'console_scripts': [
            'auto-eq-finder=automatic_eq_finder.cli.__main__:main',
        ],
    },
)