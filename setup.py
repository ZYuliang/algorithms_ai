from setuptools import setup, find_packages

setup(
    name='algorithms_ai',
    version='0.1.2',
    description=(
        ''
    ),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Yuliang Zhang',
    author_email='1137379695@qq.com',
    maintainer='Yuliang Zhang',
    maintainer_email='1137379695@qq.com',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/ZYuliang/algorithms-ai',
    install_requires=[
        "tqdm",
        "wandb",
        "loguru",
        "typing",
        "nltk"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
