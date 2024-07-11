import os

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dir_path, filename)
    with open(filename) as f:
        return [line.strip() for line in f]


# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'mimo.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'src/mimo/utils/libkdtree/pykdtree/kdtree.c',
        'src/mimo/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy.get_include()],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'mimo.utils.libmcubes.mcubes',
    sources=[
        'src/mimo/utils/libmcubes/mcubes.pyx',
        'src/mimo/utils/libmcubes/pywrapper.cpp',
        'src/mimo/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'mimo.utils.libmesh.triangle_hash',
    sources=[
        'src/mimo/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'mimo.utils.libmise.mise',
    sources=[
        'src/mimo/utils/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'mimo.utils.libsimplify.simplify_mesh',
    sources=[
        'src/mimo/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'mimo.utils.libvoxelize.voxelize',
    sources=[
        'src/mimo/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

packages = find_packages('src')
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'mimo' or p.startswith('mimo.')


def pkg_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('../..', path, filename))
    return paths


# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    # dmc_pred2mesh_module,
    # dmc_cuda_module,
]

setup(
    name="mimo",
    version="1.0.0",
    author="Yichen Cai",
    description="multi-feature implict model",
    packages=packages,
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=read_requirements_file('requirements.txt') +
                     ['airobot @ git+https://github.com/Improbable-AI/airobot.git@panda-2f140#egg=airobot'] +
                     ['urdfpy @ git+https://github.com/erwincoumans/urdfpy.git'],
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
        # 'build_ext': BuildExtension
    }
)
