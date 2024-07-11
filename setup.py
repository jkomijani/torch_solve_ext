# Copyright (c) 2024 Javad Komijani


from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


packages = ['torch_solve_ext', 'torch_solve_ext.integrate']

package_dir = {
        'torch_solve_ext': 'src',
        'torch_solve_ext.integrate': 'src/integrate'
        }

setup(name='torch_solve_ext',
      version='0.0',
      description="a package for solving (differential) equations with torch.",
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/torch_solve_ext',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      install_requires=['numpy>=2.0.0', 'torch>=2.3.1'],
      zip_safe=False
      )
