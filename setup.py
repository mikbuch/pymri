#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-


__author__ = 'Mikolaj Buchwald, mikolaj.buchwald@gmail.com'


from setuptools import setup, find_packages


setup(
    name="PyMRI",
    version="0.0.1",
    description="PyMRI is tool for MRI and fMRI data analysis.",
    license="BSD",
    keywords="MRI fMRI Machine_Learning",
    url="http://mikbuch.github.io/pymir",
    packages=find_packages(exclude=['examples', 'docs']),
    include_package_data=True,
    # test_suite='pymri.tests.runtests.make_test_suite',
    install_requires = ["scipy"],
)
