from setuptools import setup, find_packages

setup(
    name='DBV_Group06',
    version='1.0',
    python_requries='>=3.6',
    packages=find_packages(where='project'),
    package_dir={'': 'project'},
    author='Höltje, Judasz, Nörtemann',
    install_requires=[
        'numpy>=1.20.0',
        'opencv-contrib-python',

        # Possibly only required during dev phases
        'matplotlib',
        'scikit-learn',  # for static saliency

        # User Interface
        'pyside6',  # make sure shiboken6 and pyside6 have the same version
        'shiboken6',

        # NN model requirements
        'torch',
        'torchvision',
        'tqdm',

        # mostly used by the extra saliency methods imported
        'scipy>=1.6',
        'networkx>=2.5',
        'scikit-image',

    ],
    # optional requirements
    extras_require={
        # Port workloads onto graphics adapter
        # requires native nvcc installation!
        'cuda': [
            # optional requirements for the faster PatchMatch Algorithm
            'pycuda>=2021.1',
            'cupy-cuda114'
        ]
    }
)
