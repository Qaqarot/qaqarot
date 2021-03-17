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
"""Decorators."""
from keyword import iskeyword
from typing import Callable, Union

from .circuit import BlueqatGlobalSetting


def circuitmacro(func: Union[Callable, str, None] = None,
                 *,
                 allow_overwrite: bool = True) -> Callable:
    """@circuitmacro decorator.

    Typical usage:
    Case 1: no arguments

        @def_macro
        def egg(c):
            ...

    equivalent to this:

        def egg(c):
            ...
        BlueqatGlobalSetting.register_macro('egg', egg, allow_overwrite=True)


    Case 2: with name:

        @def_macro('bacon')
        def egg(c):
            ...

    is equivalent with

        def egg(c):
            ...
        BlueqatGlobalSetting.register_macro('bacon', egg, allow_overwrite=True)

    Case 3: with allow_overwrite keyword argument

        @def_macro(allow_overwrite=False)
        def egg(c):
            ...

    or

        @def_macro('bacon', allow_overwrite=False)
        def bacon(c):
            ...

    call BlueqatGlobalSetting.register_macro with allow_overwrite=False.

    Please note that `allow_overwrite=True` is default behavior.
    It is convenient for interactive environment likes Jupyter Notebook.
    However, if you're library developer, using `allow_overwrite=False` is hardly recommended.
    """
    if callable(func):
        # @def_macro pattern.
        name = func.__name__
        if not name.isidentifier() or iskeyword(name):
            raise ValueError(
                f'Function name {name} is not a valid macro name. ')
        BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
        return func
    if isinstance(func, str):
        # @def_macro(name) or @def_macro(name, allow_overwrite) pattern.
        name = func

        def _wrapper1(func):
            BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
            return func

        return _wrapper1
    if func is None:
        # @def_macro(allow_overwrite) pattern.
        def _wrapper2(func):
            name = func.__name__
            if not name.isidentifier() or iskeyword(name):
                raise ValueError(
                    f'Function name {name} is not a valid macro name. ')
            BlueqatGlobalSetting.register_macro(name, func, allow_overwrite)
            return func

        return _wrapper2
    raise TypeError('Invalid type for first argument.')
