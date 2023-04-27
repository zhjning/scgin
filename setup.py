#!/usr/bin/env python

from distutils.core import setup,Command

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'tests/runtests.py'])
        raise SystemExit(errno)

setup(name='scginpy',
      description='Python Package for Rank Aggregation',
      author='Jianing Zhang',
      author_email='zhangjn57@sysu.edu.cn',
      url='https://github.com/zhjning/scginpy',
      packages=['scginpy'],
      package_dir = {'scginpy': ''},
      package_data = {'scginpy' : ['tests/*.py']},
      cmdclass = {'test': PyTest},
      license='BSD-3',
      classifiers=[
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Programming Language :: Python',
          'Topic :: Statistics',
      ],
)