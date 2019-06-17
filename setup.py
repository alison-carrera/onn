from setuptools import setup, find_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='onn',
      version='0.0.6',
      description='Online Neural Network',
      url='https://github.com/alison-carrera/onn',
      author='Alison Carrera',
      author_email='alison.carrera2007@gmail.com',
      packages=find_packages(),
      install_requires=['numpy', 'torch'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='Apache 2.0',
      zip_safe=False)