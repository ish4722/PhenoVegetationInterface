from setuptools import setup,find_packages

setup(
    name="phenoAI",
    version="0.2",
    description="des",
    long_description="long des",
    author="Ishan_Garv_Medhansh_GarvJ_Aditi_Chehek_Niviti",
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'tensorflow',
        'keras',
        'tqdm',
        'segmentation_models',
        'xlsxwriter',
        'albumentations',
        'imgaug',
    ],
)