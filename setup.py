from setuptools import setup, find_packages

import imp

version = imp.load_source('wcqtlib.version', 'wcqtlib/version.py')

setup(
    name='wcqtlib',
    version=version.version,
    description="ISMIR2016 Library for Proving the worth of WCQTs",
    author='Christopher Jacoby, Eric J. Humphrey, Brian McFee',
    author_email='christopher.jacoby@gmail.com',
    url='https://github.com/ejhumphrey/ismir2016-wcqt',
    packages=find_packages(),
    classifiers=[
        # What does this mean?
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        # What does this mean?
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    keywords='machine learning',
    # Check/confirm.
    liscence='ISC',
    install_requires=[
        'joblib',
        'six',
        'pyzmq',
        'numpy',
        'scipy',
        'theano',
        'pandas'
    ],
    extras_require={
        'docs': ['numpydoc']
    }
)
