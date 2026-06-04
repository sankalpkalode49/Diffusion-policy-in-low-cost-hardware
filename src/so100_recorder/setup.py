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
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            
            'episode_recorder_node = so100_recorder.episode_rec:main',
            'training_node = so100_recorder.train:main',
            'telop_node = so100_recorder.teleop_rec:main',
            'servo_node = so100_recorder.servo:main',
            'ps_arm_ctl = so100_recorder.hold:main',
            'follower_node = so100_recorder.arm_follower:main',
            'diffusion_deployment_node = so100_recorder.inference:main',
            'inverse_kinematics_tester = so100_recorder.keyboard_ctl:main',
            'ik_recorder_node = so100_recorder.IK:main',
            
            
        ],
    },
)
