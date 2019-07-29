# Copyright 2019 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from blueqat import BlueqatGlobalSetting

DEFAULT_BACKEND = BlueqatGlobalSetting.get_default_backend_name()

def pytest_addoption(parser):
    parser.addoption('--add-backend', default=[DEFAULT_BACKEND], action='append')


def pytest_generate_tests(metafunc):
    if 'backend' in metafunc.fixturenames:
        metafunc.parametrize('backend', metafunc.config.getoption('--add-backend'))
