from setuptools import setup, find_packages

setup(
    name='xtrapnet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'scipy'],
    author='cyruskurd'
    author_email='cyrus.kurd@columbia.edu',
    description='A robust package for extrapolation control in neural networks',
    url='https://github.com/YourUser/xtrapnet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
