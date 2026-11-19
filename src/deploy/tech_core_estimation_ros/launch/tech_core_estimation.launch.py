from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Run the estimator on a remappable synchronized camera stream."""
    package = 'tech_core_estimation_ros'
    parameters = PathJoinSubstitution([FindPackageShare(package), 'config', 'tech_core_estimation.yaml'])
    return LaunchDescription([
        DeclareLaunchArgument('model_dir', description='Model release directory'),
        DeclareLaunchArgument('image', default_value='image'),
        DeclareLaunchArgument('camera_info', default_value='camera_info'),
        Node(
            package=package,
            executable='tech_core_estimation_node',
            parameters=[parameters, {'model_dir': ParameterValue(LaunchConfiguration('model_dir'), value_type=str)}],
            remappings=[('image', LaunchConfiguration('image')), ('camera_info', LaunchConfiguration('camera_info'))],
            output='screen',
        ),
    ])
