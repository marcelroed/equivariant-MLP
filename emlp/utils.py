import ast
import inspect
import sys
from typing import List, Iterable, Any

import torch
from joblib import Memory
from tempfile import gettempdir

# Cross-platform location for joblib memory
memory = Memory(cachedir=gettempdir(), verbose=1)
# memory.clear()


class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def _args_from_usage_string_ast(s: str) -> List[str]:
    tree = ast.parse(s)
    ast_args = tree.body[0].value.args
    args = [s[arg.col_offset:arg.end_col_offset] for arg in ast_args]
    return args


def _space_all_but_first(s: str, n_spaces: int) -> str:
    """Pad all lines except the first with n_spaces spaces"""
    lines = s.splitlines()
    for i in range(1, len(lines)):
        lines[i] = ' ' * n_spaces + lines[i]
    return '\n'.join(lines)


def _print_spaced(varnames: List[str], vals: Iterable[Any], **kwargs):
    """Print variables with their variable names"""
    for varname, val in zip(varnames, vals):
        prefix = f'{varname} = '
        n_spaces = len(prefix)
        if isinstance(val, torch.Tensor):
            val_string = str(val)
            end_of_tensor = val_string.rfind(')')
            new_val_string = val_string[:end_of_tensor] + f', shape={val.shape}' + val_string[end_of_tensor:]
            spaced = _space_all_but_first(new_val_string, n_spaces)
        else:
            spaced = _space_all_but_first(str(val), n_spaces)
        print(f'{prefix}{spaced}', **kwargs)


def dbg(*vals, **kwargs):
    """
    Print the file, linenumber, variable name and value.
    Doesn't work if expanded to multiple lines
    Eg. don't do
    ```
    dbg(
        variable
    )
    ```
    """
    frame = inspect.currentframe()
    outer_frame = inspect.getouterframes(frame)[1]

    frame_info = inspect.getframeinfo(outer_frame[0])
    string = frame_info.code_context[0].strip()

    filename = frame_info.filename.split('/')[-1]
    lineno = frame_info.lineno
    args = _args_from_usage_string_ast(string)

    # Exclude keywords arguments
    names = [arg.strip() for arg in args if '=' not in arg]

    # Prepend filename and line number
    names = [f'[{filename}:{lineno}] {name}' for name in names]

    _print_spaced(names, vals, **kwargs)
    # print('\n'.join(f'{argname} = {val}' for argname, val in zip(names, vals)))