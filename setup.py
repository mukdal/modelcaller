from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', encoding='utf-8') as f:	
    requirements = f.read().splitlines()

setup(
    name='modelcaller',
    version='0.1',
    packages=find_packages(),
    description='ModelCaller for AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mukesh Dalal',
    author_email='mukesh@aidaa.ai',
    url='https://github.com/mukdal/modelcaller',

    python_requires='>=3.9',	
    install_requires=requirements,
    
    data_files=[('', ['LICENSE.txt'])],
    license='Custom',
    license_files=('LICENSE.txt',),
    keywords='modelcaller artficial intelligence machine learning ml ai systems transformation model',

        classifiers=[	
        "Development Status :: 3 - Alpha",	
        "Intended Audience :: Developers",	
        "License :: Free for non-commercial use",	
        "Operating System :: OS Independent",	
        'Topic :: Software Development :: Libraries',	
        "Programming Language :: Python :: 3",	
        "Programming Language :: Python :: 3.12",	
    ],

)