# Created by Javad Komijani (2024)

"""This is the setup script for `torch_solve_ext`."""

from setuptools import setup


def readme():
    """Reads and returns the contents of the README.md file."""
    with open('README.md', encoding='utf-8') as f:
        return f.read()


# List of packages in the project
packages = [
    'torch_solve_ext',
    'torch_solve_ext.integrate'
]

# Dictionary of where each package resides
package_dir = {
    'torch_solve_ext': 'src',
    'torch_solve_ext.integrate': 'src/integrate'
}

setup(name='torch_solve_ext',
      version='1.0.0',
      description="a package for solving (differential) equations with torch.",
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/torch_solve_ext',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      zip_safe=False
)
