#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-


__author__ = 'Mikolaj Buchwald, mikolaj.buchwald@gmail.com'


from setuptools import setup, find_packages


setup(
    name="PyMRI",
    version="0.0.1",
    description="PyMRI is tool for MRI and fMRI data analysis.",
    license="BSD",
    keywords="MRI fMRI Machine-Learning Artificial-Intelligence",
    url="http://mikbuch.github.io/pymri",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    # test_suite='pymri.tests.runtests.make_test_suite',
    install_requires = ["scipy"],
    # this is required to force egg extraction on OS X
    # http://stackoverflow.com/a/2604694/5533240
    zip_safe=False
)
