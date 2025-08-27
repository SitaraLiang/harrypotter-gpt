"""
This script is originally from nanoGPT.

Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())
"""

import sys
from ast import literal_eval

# sys.argv[0] is the script name (train.py), so we skip it.
# We iterate over all additional arguments.
for arg in sys.argv[1:]:
    # If the argument does not contain =, it is assumed to be a Python config file.
    if '=' not in arg:
        # assume it's the name of a config file
        # Checks that it doesn’t start with -- (because --key=value arguments will be handled separately).
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())

        # executes the Python code in that file.
        # This means all variables defined in that file are now added to the current global namespace.
        exec(open(config_file).read())

    # If the argument contains = → it’s a key-value override.
    else:
        # assume it's a --key=value argument
        # Must start with --, e.g., --batch_size=32.
        assert arg.startswith('--')
        # splits into variable name (batch_size) and its new value (32).
        key, val = arg.split('=')
        # remove the leading --.
        key = key[2:]

        # Checks if the key already exists in globals() (so we don’t create a new variable by mistake).
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                # literal_eval(val) attempts to convert string to Python literal:
                #   "True" → True
                #   "3.14" → 3.14
                #   "hello" → "hello"
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # If literal_eval fails, just use the raw string.
                attempt = val

            # Checks that the type matches the original variable.
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            # Assigns the new value to globals()[key] → modifies the global variable.
            globals()[key] = attempt

        # If the key is not in globals, throw an error.
        # Prevents accidental creation of a wrong variable name from a typo.
        else:
            raise ValueError(f"Unknown config key: {key}")
