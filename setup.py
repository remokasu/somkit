from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('LICENSE', 'r', encoding='utf-8') as f:
    license = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    install_requires = f.read()

setup(
    name="pysom",
    version="0.0.1",
    url="https://github.com/remokasu/pysom",
    author="remokasu",
    description="Self-Organizing Maps (SOMs) in Python.",
    packages=find_packages(),  # 自動的にパッケージとサブパッケージを見つける
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ]
)