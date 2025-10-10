import os
from setuptools import setup, find_packages


def gen_data_files(src_dir):
    """
    generates a list of files contained in the given directory (and its
    subdirectories) in the format required by the ``package_data`` parameter
    of the ``setuptools.setup`` function.

    Parameters
    ----------
    src_dir : str
        (relative) path to the directory structure containing the files to
        be included in the package distribution

    Returns
    -------
    fpaths : list(str)
        a list of file paths
    """
    fpaths = []
    base = os.path.dirname(src_dir)
    for root, dir, files in os.walk(src_dir):
        if len(files) != 0:
            for f in files:
                fpaths.append(os.path.relpath(os.path.join(root, f), base))
    return fpaths


setup(
    name='isanlp_rst',
    version='3.1.1',
    description='IsaNLP RST Parser: A library for parsing Rhetorical Structure Theory trees.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Elena Chistova',
    author_email='elenachistov@gmail.com',
    url='https://github.com/tchewik/isanlp_rst',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    package_data = {'isanlp_rst.rstviewer': gen_data_files('isanlp_rst/rstviewer/data')},
    include_package_data=True,
    install_requires=[
        'asyncio',
        'playwright',
        'razdel',
        'fire',
        'matplotlib',
        'lxml',
        'jsonnet',
        'nltk',
        'spacy',
        'numpy==1.26.4',
        'transformers',
        'torch',
    ],
    dependency_links=[
        'git+https://github.com/iinemo/isanlp.git',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
