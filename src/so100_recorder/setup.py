from setuptools import find_packages, setup

package_name = 'so100_recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sankalp',
    maintainer_email='sankalp@todo.todo',
    description='Diffusion policy data collection, training and deployment for the SO-100 arm',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Names used in the README
            'scan = so100_recorder.scan:main',
            'teleop_rec = so100_recorder.teleop_rec:main',
            'extract_frame = so100_recorder.extract_frame:main',
            'train = so100_recorder.train:main',
            'inference = so100_recorder.inference:main',
            # Original names, kept for compatibility
            'episode_recorder_node = so100_recorder.episode_rec:main',
            'training_node = so100_recorder.train:main',
            'telop_node = so100_recorder.teleop_rec:main',
            'diffusion_deployment_node = so100_recorder.inference:main',
        ],
    },
)
