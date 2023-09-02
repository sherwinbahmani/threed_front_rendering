""" All default values are stored here. """


class DefaultConfig:
    """
    All the default config values are specified in this class.
    """
    # Camera
    resolution_x = 256 #512
    resolution_y = 256 #512
    clip_start = 0.1
    clip_end = 1000
    fov = 0.691111
    pixel_aspect_x = 1
    pixel_aspect_y = 1
    shift_x = 0
    shift_y = 0
    lens_unit = "FOV"

    # Stereo
    stereo_convergence_mode = "PARALLEL"
    stereo_convergence_distance = 0.00001
    stereo_interocular_distance = 0.065

    # Renderer
    file_format = "PNG"
    color_depth = 8
    enable_transparency = False
    jpg_quality = 95
    samples = 1024
    sampling_noise_threshold = 0.01
    cpu_threads = 1
    denoiser = "INTEL"
    simplify_subdivision_render = 3
    diffuse_bounces = 0 #3
    glossy_bounces = 0
    ao_bounces_render = 0 #3
    max_bounces = 0 #3
    transmission_bounces = 0
    transparency_bounces = 0 #8
    volume_bounces = 0
    antialiasing_distance_max = 10000
    # world_background = [0.05, 0.05, 0.05]
    # world_background = [1.00, 1.00, 1.00]
    # world_background = [1.50, 1.50, 1.50]
    world_background = [1.75, 1.75, 1.75]
    # world_background = [5.00, 5.00, 5.00]
    # world_background = [2.00, 2.00, 2.00]
    # world_background = [20.00, 20.00, 20.00]
    # world_background = [10.00, 10.00, 10.00]
    # world_background = [255.00, 255.00, 255.00]

    # Setup
    default_pip_packages = ["wheel", "pyyaml==5.1.2", "imageio==2.9.0", "gitpython==3.1.18",
                            "scikit-image==0.19.2", "pypng==0.0.20", "scipy==1.7.3", "matplotlib==3.5.1",
                            "pytz==2021.1", "h5py==3.6.0", "Pillow==8.3.2", "opencv-contrib-python==4.5.5.64",
                            "scikit-learn==1.0.2", "python-dateutil==2.8.2", "rich==12.6.0", "pyrr"]
