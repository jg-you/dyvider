import time
import datetime
from setuptools import setup

# Version information
name = 'dyvider'
major = "0"
minor = "3"

description="Rapid and exact partitioning algorithms for graphs embedded in one dimension."

license = 'MIT'
authors = "Jean-Gabriel Young, Alice Patania"
author_email = "jean-gabriel.young@uvm.edu"
url = 'https://github.com/jg-you/dyvider'
download_url = 'https://pypi.python.org/pypi/dyvider/'
platforms = ['Linux', 'Unix']
keywords = ['graphs', 'community detection', 'networks', 'inference']
classifiers = [
    'Development Status :: 4 - Beta  ',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License  ',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Scientific/Engineering :: Mathematics']

date_info = datetime.datetime.utcfromtimestamp(int(time.time()))
date = time.asctime(date_info.timetuple())
___version___ = major + "." + minor


setup(name=name,
      version=___version___,
      description=description,
      long_description=open('long_description.rst').read(),
      url=url,
      author=authors,
      author_email=author_email,
      classifiers=classifiers,
      keywords=keywords,
      license=license,
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'networkx',
          'numpy'
      ])