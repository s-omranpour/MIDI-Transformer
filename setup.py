from setuptools import setup

url = 'https://github.com/s-omranpour/MIDI-Transformer'

setup(
    name='midi_transformer',
    version='0.0.1',
    description='generate multi instrument music',
    url=url,
    author='Soroush Omranpour',
    author_email='soroush.333@gmail.com',
    keywords='midi, deep learning, music',
    packages=[
        'midi_transformer', 
        'midi_transformer.modules'
    ],
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy',
        'torch',
        'pytorch_lightning',
        'pytorch_fast_transformers',
        'deepnote >= 0.0.12'
    ],
    license="MIT license",

    project_urls={
        'Bug Reports': url + '/issues',
        'Source': url,
    },
)