import json
import multiprocessing
import os
import sys


class CallableObject:
    """A small callable data structure that contains 4 variables, two of which are usually inited and the third is later set."""

    def __init__(self, location, name, callable_object=None, *argument, **keyword_argument):
        """
        Initialize the CallableObject.

        Args:
            location (str): The location of the module containing the callable.
            name (str): The name of the callable function.
            callable_object (callable, optional): The callable function. Defaults to None.
            argument: Arguments to pass to the callable.
            keyword_argument: Keyword arguments to pass to the callable.
        """
        self.location = location
        self.name = name
        self.arg = argument
        self.kwargs = keyword_argument
        self.callable_object = callable_object


class PythonExecuter:
    """A class to execute Python functions in a specified order."""

    def __init__(self, path_to_file=None, config_file=None, config_list=None, context_list=None):
        """
        Initialize the PythonExecuter.

        Args:
            path_to_file (str, optional): The path to the file containing the configuration. Defaults to None.
            config_file (str, optional): The name of the configuration file. Defaults to None.
            config_list (dict, optional): A pre-parsed configuration dictionary. Defaults to None.
            context_list (dict, optional): A dictionary containing context variables. Defaults to None.
        """
        self.path = path_to_file
        self.config_list = config_list
        self.config_file = config_file
        self.context_list = context_list
        self.callable_objects = []

    def load_statements(self):
        """Load statements from a file or a pre-parsed configuration dictionary."""
        try:
            if self.config_file is not None:
                with open(os.path.join(self.path, self.config_file), 'r') as json_file:
                    dict_data = json.load(json_file)
            else:
                dict_data = self.config_list

            d_a = dict_data["algorithm"]
            tmp = CallableObject(
                dict_data["algorithm_path"], d_a["file"] + '.' + d_a["callable"], None, *d_a["arguments"],
                **d_a["keyword_arguments"])
            self.callable_objects.append(tmp)
        except (FileNotFoundError, KeyError, TypeError) as e:
            raise Exception(f"Failed to load configuration: {str(e)}")

    def make_higher_order(self, string=None):
        """
        Evaluate a function name or import a module and evaluate a function.

        Args:
            string (str, optional): A function name or an import string. Defaults to None.

        Returns:
            callable: The evaluated callable function.
        """
        try:
            if string is not None:
                if '.' in string:
                    for i in self.callable_objects:
                        if i.name == string:
                            return i.callable_object
                    if i.location is not None:
                        sys.path.append(i.location)
                    module_string = string.split('.')[0]
                    exec(f'import {module_string}')
                    return eval(string)
                else:
                    return eval(string)
            else:
                for i in self.callable_objects:
                    if '.' in i.name:
                        if i.location is not None:
                            sys.path.append(i.location)
                            print(i.location)
                        module_string = i.name.split('.')[0]
                        exec(f'import {module_string}')
                    i.callable_object = eval(i.name)
        except (ModuleNotFoundError, AttributeError, TypeError) as e:
            raise TypeError('The given callable object is not actually callable.')

    def wrap_function(self, func: callable, *args, **kwargs):
        '''Take a callable and an arbitrary number of arguments and keyword arguments and return an evaluated callable.'''

        return func(*args, **kwargs)

    def execute_functions_pool(self):
        '''Execute the given functions in order of input, using multiprocessing pool.'''

        with multiprocessing.get_context('spawn').Pool(processes=1) as pool:
            result = pool.apply(self.execute_functions, [])
        return result

    def execute_functions(self):
        '''Execute the given functions in order of input.'''
        results = []
        for i in self.callable_objects:
            combo = i.kwargs.copy()
            combo.update(self.context_list)
            try:
                result = self.wrap_function(i.callable_object, *i.arg, **combo)
                results.append(result)
            except Exception as e:
                print("An exception has occurred: ", e)


class CallableObject:
    """A small callable data structure that contains 4 variables, of which usually two are initialized and the third is later set."""

    def __init__(self, location, name, callable_object=None, *args, **kwargs):
        """Initialize the callable object with the given location, name, arguments and keyword arguments.

        Args:
            location (str): The location of the callable object.
            name (str): The name of the callable object.
            callable_object (callable, optional): The callable object. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            TypeError: If the given callable object is not actually callable.
        """
        self.location = location
        self.name = name
        self.arg = args
        self.kwargs = kwargs
        self.callable_object = callable_object
        if callable_object is not None and not callable(callable_object):
            raise TypeError('The given callable object is not actually callable.')
