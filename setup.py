from setuptools import setup, find_packages

setup(
    name='kvae_torch',
    version='0.1.0',
    description='KVAE implementation with PyTorch',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'torch',
        'lightning',
        'tqdm',
        'matplotlib',
        'seaborn',
    ],
    extras_require={
        'dev': ['pytest'],
    },
    include_package_data=True,
)
