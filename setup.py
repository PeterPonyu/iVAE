from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name = 'iVAE',
    version = '0.0.2',
    description = 'Interpretable Variational Autoencoder for Single-Cell Data',
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    url = 'https://github.com/PeterPonyu/iVAE',
    author = 'Zeyu Fu',
    author_email = 'fuzeyu99@126.com',
    license = 'MIT',
    packages = find_packages(),
    install_requires = [
        'scanpy>=1.10.4',
        'scib>=1.1.5',
        'torch>=1.13.1',
        'joblib>=1.2.0',
        'tqdm>=4.64.1',
    ],
    entry_points={
        'console_scripts': [
            'iVAE=iVAE.cli:main',
        ]
    },
    package_data={
        'iVAE': ['fig.png'],
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    python_requires='>=3.9',
)
