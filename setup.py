import setuptools

setuptools.setup(
    name='libscrna',
    version='0.4.0',
    author='Antony B Holmes',
    author_email='antony.b.holmes@gmail.com',
    description='A library for dealing with single cell RNA-seq data.',
    url='https://github.com/antonybholmes/libscrna',
    packages=setuptools.find_packages(),
    install_requires=[
          'libplot',
          'libsparse',
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
)
