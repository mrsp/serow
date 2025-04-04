from setuptools import setup, find_packages, Extension
import os
from setuptools.command.build_ext import build_ext
import subprocess
import sys

# Get the directory where setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Get the source directory based on the extension name
        if ext.name == 'serow.contact_ekf':
            sourcedir = os.path.join(setup_dir, 'serow')
        elif ext.name == 'serow.state':
            sourcedir = os.path.join(setup_dir, 'serow')
        else:
            raise ValueError(f"Unknown extension: {ext.name}")

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_PREFIX_PATH={os.path.join(setup_dir, "..", "core")}'
        ]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        subprocess.check_call(['cmake', sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build_temp)

setup(
    name="serow",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=[
        Extension('serow.contact_ekf', sources=[], extra_compile_args=[], extra_link_args=[]),
        Extension('serow.state', sources=[], extra_compile_args=[], extra_link_args=[]),
    ],
    cmdclass=dict(build_ext=CMakeBuild),
    install_requires=[
        "numpy",
        "pybind11",
    ],
    python_requires=">=3.6",
) 