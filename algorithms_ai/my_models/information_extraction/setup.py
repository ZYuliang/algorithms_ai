from setuptools import setup, find_packages

setup(
    name='information_extraction',
    version='0.1.0',
    description=(
        ''
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yuliang Zhang',
    author_email='',
    maintainer='Yuliang Zhang',
    maintainer_email='',
    packages=find_packages(),
    platforms=["all"],
    url='',
    install_requires=[
        "numpy==1.26.1",
        "transformers==4.35.0",
        "loguru==0.6.0",
        'accelerate==0.24.1',
        "datasets==2.14.6",
        "wandb==0.16.0",
        "torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116"
    ],
    extras_require={
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
