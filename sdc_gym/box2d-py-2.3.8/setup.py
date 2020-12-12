#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for pybox2d.

For installation instructions, see INSTALL.

Basic install steps:
 python setup.py build

If that worked, then:
 python setup.py install
"""

import os
import sys
from glob import glob

__author__='Ken Lauer'
__license__='zlib'
__date__="$Date$"
__version__="$Revision$"

import setuptools
from setuptools import (setup, Extension)
from setuptools.command.install import install

setuptools_version = setuptools.__version__
print('Using setuptools (version %s).' % setuptools_version)

if setuptools_version:
    if (setuptools_version in ["0.6c%d"%i for i in range(1,9)] # old versions
        or setuptools_version=="0.7a1"): # 0.7a1 py 3k alpha version based on old version
            print('Patching setuptools.build_ext.get_ext_filename')
            from setuptools.command import build_ext
            def get_ext_filename(self, fullname):
                from setuptools.command.build_ext import (_build_ext, Library, use_stubs)
                filename = _build_ext.get_ext_filename(self,fullname)
                if fullname in self.ext_map:
                    ext = self.ext_map[fullname]
                    if isinstance(ext,Library):
                        fn, ext = os.path.splitext(filename)
                        return self.shlib_compiler.library_filename(fn,libtype)
                    elif use_stubs and ext._links_to_dynamic:
                        d,fn = os.path.split(filename)
                        return os.path.join(d,'dl-'+fn)
                return filename
            build_ext.build_ext.get_ext_filename = get_ext_filename

# release version number
box2d_version  = '2.3'
release_number = 8

# create the version string
version_str = "%s.%s" % (box2d_version, release_number)

# setup some paths and names
library_base='library' # the directory where the egg base will be for setuptools develop command
library_name='Box2D'   # the final name that the library should end up being
library_path=os.path.join(library_base, library_name) # library/Box2D (e.g.)
source_dir='Box2D' # where all of the C++ and SWIG source resides
swig_source='Box2D.i' # the main SWIG source file
use_kwargs=True # whether or not to default creating kwargs for all functions


def write_init():
    # read in the license header
    license_header = open(os.path.join(source_dir, 'pybox2d_license_header.txt')).read()

    # create the source code for the file
    if sys.version_info >= (2, 6):
        import_string = "from .%s import *" % library_name
    else:
        import_string = "from %s import *" % library_name

    init_source = [
        import_string,
        "__author__ = '%s'" % __date__ ,
        "__version__ = '%s'" % version_str,
        "__version_info__ = (%s,%d)" % (box2d_version.replace('.', ','), release_number),
        "__revision__ = '%s'" % __version__,
        "__license__ = '%s'" % __license__ ,
        "__date__ = '%s'" % __date__ , ]

    # and create the __init__ file with the appropriate version string
    f=open(os.path.join(library_path, '__init__.py'), 'w')
    f.write(license_header)
    f.write( '\n'.join(init_source) )
    f.close()

source_paths = [
    os.path.join(source_dir, 'Dynamics'),
    os.path.join(source_dir, 'Dynamics', 'Contacts'),
    os.path.join(source_dir, 'Dynamics', 'Joints'),
    os.path.join(source_dir, 'Common'),
    os.path.join(source_dir, 'Collision'),
    os.path.join(source_dir, 'Collision', 'Shapes'),
    ]

# glob all of the paths and then flatten the list into one
box2d_source_files = [os.path.join(source_dir, swig_source)] + \
    sum( [glob(os.path.join(path, "*.cpp")) for path in source_paths], [])

# arguments to pass to SWIG. for old versions of SWIG, -O (optimize) might not be present.
# Defaults:
# -O optimize, -includeall follow all include statements, -globals changes cvar->b2Globals
# -Isource_dir adds source dir to include path, -outdir library_path sets the output directory
# -small makes the Box2D_wrap.cpp file almost unreadable, but faster to compile. If you want
# to try to understand it for whatever reason, I'd recommend removing that option.
swig_arguments = \
        '-c++ -I%s -small -O -includeall -ignoremissing -w201 -globals b2Globals -outdir %s' \
        % (source_dir, library_path)

if use_kwargs:
    # turn off the warnings about functions that can't use kwargs (-w511)
    # and let the wrapper know we're using kwargs (-D_SWIG_KWARGS)
    swig_arguments += " -keyword -w511 -D_SWIG_KWARGS"

# depending on the platform, add extra compilation arguments. hopefully if the platform
# isn't windows, g++ will be used; -Wno-unused then would suppress some annoying warnings
# about the Box2D source.
if sys.platform in ('win32', 'win64'):
    extra_args=['-I.']
else:
    extra_args=['-I.', '-Wno-unused']

pybox2d_extension = \
    Extension('Box2D._Box2D', box2d_source_files, extra_compile_args=extra_args, language='c++')

LONG_DESCRIPTION = \
""" 2D physics library Box2D %s for usage in Python.

    After installing please be sure to try out the testbed demos.
    They require either pygame or pyglet and are available on the
    homepage.

    pybox2d homepage: http://pybox2d.googlecode.com
    Box2D homepage: http://www.box2d.org
    """ % (box2d_version,)

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: zlib/libpng License",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Libraries :: pygame",
    ]

class PostInstallHook(install):
    '''
    Describes actions to be executed after the install. 
    In this case, install has the wrong order of operations -
    it first copies *.py files, then builds *.py files using swig.
    This class copies new *.py files one more time after install. 
    '''
    def run(self):
        import os
        import shutil
        recopy_list = ['Box2D.py', '__init__.py']

        install.run(self)
    
        dest = os.path.split(self.get_outputs()[0])[0]
        for fname in recopy_list:
            local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), library_base, library_name, fname)
            installed_path = os.path.join(dest, fname)

            print('Re-copying {} --> {}'.format(local_path, installed_path))
            shutil.copyfile(local_path, installed_path)


write_init()

setup_dict = dict(
    name             = "box2d-py",
    version          = version_str,
    author           = "Ken Lauer",
    author_email     = "sirkne@gmail.com",
    description      = "Python Box2D",
    license          = "zlib",
    url              = "https://github.com/openai/box2d-py",
    long_description = LONG_DESCRIPTION,
    classifiers      = CLASSIFIERS,
    packages         = ['Box2D', 'Box2D.b2'],
    package_dir      = {'Box2D': library_path,
                        'Box2D.b2': os.path.join(library_path, 'b2'),
                        'Box2D.tests' : 'tests'},
    test_suite       = 'tests',
    options          = { 'build_ext': { 'swig_opts' : swig_arguments },
                         'egg_info' : { 'egg_base' : library_base },
                        },
    ext_modules      = [ pybox2d_extension ],
    cmdclass         = {'install': PostInstallHook},

#   use_2to3         = (sys.version_info >= (3,)),
    )

# run the actual setup from distutils
setup(**setup_dict)
