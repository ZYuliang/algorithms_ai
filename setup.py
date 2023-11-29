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
        "tqdm==4.65.0",
        "loguru==0.6.0",
    ],
    extras_require={
        'elastic_search_utils': ['elasticsearch==8.8.2', ],
        'kafka_utils': ['aiokafka==0.8.1', 'six==1.16.0', 'pydantic==1.8.2'],
        'dataset_utils':['']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
