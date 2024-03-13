from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='modelcaller',
    version='0.1',
    packages=find_packages(),
    description='Intermediary objects for calling AI models',
    author='Mukesh Dalal',
    author_email='mukesh@aidaa.ai',
    
    data_files=[('', ['LICENSE.txt'])],
    license='Custom',
    license_files=('LICENSE.txt',),

    long_description=long_description,
    long_description_content_type='text/markdown',
)