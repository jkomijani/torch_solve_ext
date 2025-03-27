# Created by Javad Komijani (2024)

from setuptools import setup


def readme():
    """Read the contents of the README file."""
    with open('README.rst') as f:
        return f.read()


# List of packages in the project
packages = [
    'torch_solve_ext',
    'torch_solve_ext.integrate',
    'torch_solve_ext.stats'
]

# Dictionary of where each package resides
package_dir = {
    'torch_solve_ext': 'src',
    'torch_solve_ext.integrate': 'src/integrate',
    'torch_solve_ext.stats': 'src/stats'
}

setup(name='torch_solve_ext',
      version='1.0',
      description="a package for solving (differential) equations with torch.",
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/torch_solve_ext',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      zip_safe=False
)
