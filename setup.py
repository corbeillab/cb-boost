# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2021 Baptiste BAUVIN
# -----------------
#
# Licensed under the New BSD license,  (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Contributors:
# ------------
#
# * Baptiste Bauvin <baptiste.bauvin.1@ulaval.ca>
#
#
# Description:
# -----------
# Description: C-Bound boosting algorithm
#
#
# Version:
# -------
# Version: 0.0
#
#
# Licence:
# -------
# New BSD
#
#
# ######### COPYRIGHT #########
import os
from setuptools import setup, find_packages

def setup_package():
    """
    Setup function
    """
    name = 'cb_boost'
    version = 0.0
    dir = 'cb_boost'
    description = 'Scalable C-Bound boosting algorithm'
    here = os.path.abspath(os.path.dirname(__file__))
    url = "https://github.com/babau1/cb_boost"
    project_urls = {
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Baptiste BAUVIN'
    author_email = 'baptiste.bauvin@lis-lab.fr'
    maintainer = 'Baptiste BAUVIN',
    maintainer_email = 'baptiste.bauvin@lis-lab.fr',
    license = 'New BSD'
    keywords = ('machine learning, supervised learning, classification, '
                'ensemble methods, c-bound')
    packages = find_packages(exclude=['*.tests'])
    install_requires = ['scikit-learn>=0.19', 'numpy', 'scipy', 'logging']
    python_requires = '>=3.5'
    extras_require = {}
    include_package_data = True

    setup(name=name,
          version=version,
          description=description,
          url=url,
          project_urls=project_urls,
          author=author,
          author_email=author_email,
          maintainer=maintainer,
          maintainer_email=maintainer_email,
          license=license,
          keywords=keywords,
          packages=packages,
          install_requires=install_requires,
          python_requires=python_requires,
          extras_require=extras_require,
          include_package_data=include_package_data)

if __name__ == "__main__":
    setup_package()
