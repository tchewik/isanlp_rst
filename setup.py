from setuptools import setup, find_packages


setup(
    name='isanlp_rst',
    version='3.0.1a',
    description='IsaNLP RST Parser: A library for parsing Rhetorical Structure Theory trees.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Elena Chistova',
    author_email='elenachistov@gmail.com',
    url='https://github.com/tchewik/isanlp_rst',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=[
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
