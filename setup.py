import os
import glob
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.cmd import Command


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        install.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    PY_CLEAN_FILES = ['./build', './dist', './__pycache__', './*.pyc', './*.tgz', './*.egg-info', './.eggs']
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root_dir = os.path.dirname(os.path.realpath(__file__))
        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(root_dir, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(root_dir):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, root_dir))
                print('Removing %s' % os.path.relpath(path))
                shutil.rmtree(path)

setup(
    name='volksdep',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='hxcai',
    author_email='chxlll@126.com',
    description='An easy toolbox for deploying and accelerating pytorch model with tensorRT in x86 and arch64 platform.',
    long_description=read('README.md'),
    setup_requires=[
        'tensorrt>=6.0.1.0',
    ],
    install_requires=[
        'numpy',
        'onnx<=1.6.0',
        'pycuda>=2019.1.2',
        'torch==1.2.*',
    ],
    python_requires='>3.6',
    cmdclass={
        'clean': CleanCommand,
    },
    zip_safe=False,
)
