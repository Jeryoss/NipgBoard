from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys
import glob
import threading
from datetime import datetime

from cryptography.fernet import Fernet

import tensorflow as tf
from werkzeug import wrappers

from tensorboard.backend.http_util import Respond
from tensorboard.plugins import base_plugin
from tensorboard import program

_PLUGIN_PREFIX_ROUTE = 'multidash'
# _PLUGIN_NAME = 'org_tensorflow_tensorboard_multidash'
# _PLUGINS_DIR = 'plugins'

RUNS_ROUTE = '/runs'
CNF_ROUTE = '/cnf'
LAYOUT_ROUTE = '/layout'
LOGIN_ROUTE = '/login'
REGISTER_ROUTE = '/register'
NEEDLOGIN_ROUTE = '/needlogin'
ENCRYPT_ROUTE = '/encrypt'
BOARD_ROUTE = '/board'
DECRYPT_ROUTE = '/decrypt'
TESTEXIT_ROUTE = '/testexit'
FORGOTPASSWORD_ROUTE = '/forgotpassword'
ADDNOTIFICATION_ROUTE = '/setnotification'
GETNOTIFICATIONS_ROUTE = '/getnotifications'


class MultidashPlugin(base_plugin.TBPlugin):
    """A TensorBoard plugin that replaces the default tab-based
    view with a customizable window layout.

    Attributes:
        plugin_name (str): The name of the plugin's route prefix.
        multiplexer: The event multiplexer instance.
        logdir (str): The directory containing event files.
        _handlers (Dict[str, Callable[[Any], Response]]): A dictionary of
            route handlers.
        readers (Dict[str, tf.compat.v1.summary.FileWriter]): A dictionary of
            file writers for event files.
        run_paths: A list of run paths in the event multiplexer.
        _configs: A dictionary of configuration values for the plugin.
        old_num_run_paths: The number of run paths in the multiplexer when the
            plugin was last active.
        config_fpaths: The file paths of configuration files for the plugin.
        _is_active: A flag indicating whether the plugin is active.
        _thread_for_determining_is_active: A thread for determining whether
            the plugin is active.
    """

    plugin_name = _PLUGIN_PREFIX_ROUTE

    def __init__(self, context):
        self.multiplexer = context.multiplexer
        self.logdir = context.logdir
        self._handlers = None
        self.readers = {}
        self.run_paths = None
        self._configs = None
        self.old_num_run_paths = None
        self.config_fpaths = None
        self._is_active = False
        self._thread_for_determining_is_active = None
        if self.multiplexer:
            self.run_paths = self.multiplexer.RunPaths()

        self.key = b'vei8dcIyZU-XeMiydKm6DVRJSjhjgQ_bA0h6pqHD1Q4='
        self.f = Fernet(self.key)

    def get_plugin_apps(self):
        self._handlers = {
            RUNS_ROUTE: self._serve_runs,
            CNF_ROUTE: self._serve_cnf,
            LAYOUT_ROUTE: self._save_layout,
            LOGIN_ROUTE: self._serve_login,
            REGISTER_ROUTE: self._serve_register,
            NEEDLOGIN_ROUTE: self._serve_needlogin,
            ENCRYPT_ROUTE: self._serve_encrypt,
            BOARD_ROUTE: self._serve_boardpath,
            DECRYPT_ROUTE: self._serve_decrypt,
            TESTEXIT_ROUTE: self._serve_testexit,
            FORGOTPASSWORD_ROUTE: self._serve_forgotpassword,
            ADDNOTIFICATION_ROUTE: self._serve_addnotification,
            GETNOTIFICATIONS_ROUTE: self._serve_notifications,
        }
        return self._handlers

    def is_active(self) -> bool:
        """Determines whether the plugin is active (has meaningful data to process and serve).

        Returns:
            A boolean indicating whether the plugin is active.
        """
        if self._is_active:
            # The plugin has already been determined to be active.
            return True
        elif self._thread_for_determining_is_active:
            # Another thread is already determining whether the plugin is active.
            return False
        else:
            # Start a new thread to determine whether the plugin is active.
            self._thread_for_determining_is_active = threading.Thread(
                target=self._determine_is_active
            )
            self._thread_for_determining_is_active.start()
            return False

    def _determine_is_active(self):
        """
        Determines whether the plugin is active.

        This method is run in a separate thread so that the plugin can offer an
        immediate response to whether it is active and determine whether it should
        be active in a separate thread.
        """
        if self.configs:
            self._is_active = True
        self._thread_for_determining_is_active = None

    @property
    def configs(self):
        """
        Returns the configs.
        """
        return self._configs

    def _run_paths_changed(self):
        """
        Returns whether the run paths have changed since the last time this method
        was called.
        """
        num_run_paths = len(list(self.run_paths.keys()))
        if num_run_paths != self.old_num_run_paths:
            self.old_num_run_paths = num_run_paths
            return True
        return False

    def _get_reader_for_run(self, run):
        """
        Returns a checkpoint reader for a given run.

        Args:
          run: The name of the run.

        Returns:
          A checkpoint reader for the given run.
        """
        if run in self.readers:
            return self.readers[run]

        config = self._configs[run]
        reader = None
        if config.model_checkpoint_path:
            try:
                reader = tf.pywrap_tensorflow.NewCheckpointReader(
                    config.model_checkpoint_path)
            except Exception as e:
                tf.logging.warning('Failed reading "%s": %s',
                                   config.model_checkpoint_path, str(e))
        self.readers[run] = reader
        return reader

    @wrappers.Request.application
    def _serve_runs(self, request):
        """Returns a list of runs that have embeddings.

        Args:
            request: The request object.

        Returns:
            A list of runs that have embeddings in JSON format.
        """
        keys = ['.']
        return Respond(request, list(keys), 'application/json')

    @wrappers.Request.application
    def _serve_cnf(self, request):
        """
        Serves the configuration file for the current or subfolder logging directory
        to be used by multidash for layout customization. Returns a JSON file containing
        plugin configuration and layout information.

        Args:
            request: The HTTP request object.

        Returns:
            An HTTP response containing the configuration file in JSON format.
        """
        config_file = "plugins.json"
        layout_file = "layout.json"
        subfolder = request.args.get('subfolder', '').strip()
        try:
            if subfolder:
                logdir = os.path.join(self.logdir, subfolder)
                if not os.path.isfile(os.path.join(logdir, config_file)):
                    return Respond(request, "Plugin configuration file not found!", "text/plain", 500)
            else:
                logdir = self.logdir
                if not os.path.isfile(os.path.join(logdir, config_file)):
                    return Respond(request, "Plugin configuration file not found!", "text/plain", 500)

            with tf.gfile.GFile(os.path.join(logdir, config_file), 'r') as json_file:
                try:
                    self.cnf_json = json.load(json_file)
                    if os.path.isfile(os.path.join(logdir, layout_file)):
                        with tf.gfile.GFile(os.path.join(logdir, layout_file), 'r') as layout_json:
                            try:
                                self.cnf_json["layout"] = json.load(layout_json)
                            except json.JSONDecodeError:
                                self.cnf_json["layout"] = {}
                    else:
                        self.cnf_json["layout"] = {}
                    self.cnf_json["mode"] = program.multidash_mode
                    return Respond(request, json.dumps(self.cnf_json), 'application/json', 200)
                except json.JSONDecodeError:
                    return Respond(request, "Invalid plugin configuration file!", "text/plain", 500)
        except Exception as e:
            # Log the exception and return an error response
            print("Error serving configuration file: %s", str(e))
            return Respond(request, "Internal Server Error", "text/plain", 500)

    @wrappers.Request.application
    def _save_layout(self, request):
        """Saves the layout configuration to a JSON file.

        Args:
            request: A `Request` object.

        Returns:
            A `Respond` object with a status code of 200 if successful, or a
            `Respond` object with a status code of 500 if an error occurred.
        """
        subfolder = request.args.get('subfolder')
        try:
            layout_data = request.get_data()
            layout_str = layout_data.decode('utf-8') if isinstance(layout_data, bytes) else layout_data

            if subfolder == ' ':
                layout_file = os.path.join(self.logdir, 'layout.json')
            else:
                layout_file = os.path.join(self.logdir, subfolder, 'layout.json')

            with tf.gfile.GFile(layout_file, 'w') as f:
                f.write(layout_str)
                return Respond(request, 'OK', 'text/plain', 200)
        except Exception as e:
            print('Failed to save layout: %s', e)
            return Respond(request, str(e), 'text/plain', 500)

    @wrappers.Request.application
    def _serve_needlogin(self, request):
        """
        Checks whether the configuration file contains a "password" key and its value. Returns "yes" if the key is present,
        otherwise "no".

        Args:
            request: A Flask request object.

        Returns:
            A Flask response object containing either "yes" or "no" as a string, along with the appropriate HTTP status code.

        Raises:
            HTTPException: If an error occurs while processing the request.
        """
        try:
            config_file = os.path.join(self.logdir, 'cnf.json')
            with open(config_file) as json_file:
                data = json.load(json_file)
                if 'password' in data['default']:
                    return Respond(request, 'yes', "text/plain", 200)
                else:
                    return Respond(request, 'no', "text/plain", 200)
        except (IOError, KeyError, json.JSONDecodeError) as e:
            return Respond(request, f"Error getting notifications: {str(e)}", 'text/plain', 500)
            # raise werkzeug.exceptions.HTTPException(description='Error while processing request.')

    @wrappers.Request.application
    def _serve_notifications(self, request):
        """
        Returns notifications for the given sublogdir in JSON format.

        Args:
            request: Request object.

        Returns:
            A Response object with JSON data if notification file exists, else an error message.
        """
        try:
            sublogdir = request.args.get('sublogdir')
            notification_file = os.path.join(self.logdir, sublogdir, 'notifications.json')

            if os.path.isfile(notification_file):
                with open(notification_file) as json_file:
                    data = json.load(json_file)
                    return Respond(request, json.dumps(data), 'application/json', 200)
            else:
                return Respond(request, "Notification file not found.", 'text/plain', 404)
        except Exception as e:
            return Respond(request, f"Error getting notifications: {str(e)}", 'text/plain', 500)

    @wrappers.Request.application
    def _serve_addnotification(self, request):
        """Adds a new notification to the notifications.json file.

        Args:
            request (Request): A request object containing the notification details.

        Returns:
            A response indicating whether the notification was successfully saved.

        Raises:
            ValueError: If any of the required parameters are missing.

        """
        try:
            # Get notification details from request
            title = request.args.get('title')
            plugin = request.args.get('plugin')
            icon = request.args.get('icon')
            sublogdir = request.args.get('sublogdir')
            if not all([title, plugin, icon, sublogdir]):
                raise ValueError("Missing required parameters.")

            # Create notification dictionary
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            if icon == 'check':
                noti = {'title': title, 'icon': icon, 'date': dt_string, 'plugin': plugin, 'color': 'darkgreen'}
            elif icon == 'close':
                noti = {'title': title, 'icon': icon, 'date': dt_string, 'plugin': plugin, 'color': 'darkred'}
            else:
                noti = {'title': title, 'icon': icon, 'date': dt_string, 'plugin': plugin, 'color': 'orange'}

            # Save notification to notifications.json file
            notification_file = os.path.join(self.logdir, sublogdir, 'notifications.json')
            if os.path.isfile(notification_file):
                with open(notification_file, 'r') as json_file:
                    data = json.load(json_file)
                data.append(noti)
                while len(data) > 20:
                    del data[0]
            else:
                data = [noti]
            with open(notification_file, 'w') as fp:
                json.dump(data, fp)
            return Respond(request, 'saved', "text/plain", 200)

        except ValueError as e:
            return Respond(request, str(e), "text/plain", 400)

        except Exception as e:
            # Log any unexpected exceptions
            print("Unexpected exception in _serve_addnotification")
            return Respond(request, "An unexpected error occurred.", "text/plain", 500)

    @wrappers.Request.application
    def _serve_login(self, request):
        """Handles login requests and authenticates user.

        Args:
            request: A request object containing user's login credentials.

        Returns:
            A response object containing the user's foldername on successful login,
            and 'incorrect' message on incorrect login credentials.

        Raises:
            IOError: An error occurred when reading cnf.json file.
        """

        try:
            # Config file
            config_file = os.path.join(self.logdir, 'cnf.json')
            with open(config_file) as json_file:
                data = json.load(json_file)
        except IOError as e:
            return Respond(request, 'An error occurred when reading cnf.json file.', "text/plain", 500)

        if ('registered' not in data['default']):
            return Respond(request, 'incorrect', "text/plain", 200)

        # Query parameters
        name = request.args.get('name')
        password = request.args.get('password')

        key = 'vei8dcIyZU-XeMiydKm6DVRJSjhjgQ_bA0h6pqHD1Q4='
        f = Fernet(key)

        # check admin login
        if 'admin_password' in data['default']:
            admin_password = data['default']['admin_password']
            encoded_admin_password = admin_password.encode()
            decrypted_admin_password = f.decrypt(encoded_admin_password)
            decrypted_admin_password = decrypted_admin_password.decode("utf-8")

            if password == decrypted_admin_password:
                for i in data['default']['registered']:
                    if name == i['foldername']:
                        return Respond(request, i['foldername'], "text/plain", 200)

        # check user login
        found = False

        for i in data['default']['registered']:
            # looking for username
            if i['name'] == name:
                found = True
                correct_pw = i['password']
                encoded_password = correct_pw.encode()
                decrypted_password = f.decrypt(encoded_password)
                decrypted_password = decrypted_password.decode("utf-8")
                # check pw
                if decrypted_password == password:
                    return Respond(request, i['foldername'], "text/plain", 200)
                else:
                    return Respond(request, 'incorrect', "text/plain", 200)

        if found == False:
            return Respond(request, 'incorrect', "text/plain", 200)

    @wrappers.Request.application
    def _serve_register(self, request):
        """Handle registration form submissions.

            Args:
                request: The HTTP request object.

            Returns:
                The HTTP response object.

            Raises:
                Http404: If the requested page does not exist.
        """

        if not os.path.exists(os.path.join(self.logdir, 'cnf.json')):
            return Respond(request, 'no_config', "text/plain", 200)

        # Config file
        config_file = os.path.join(self.logdir, 'cnf.json')
        try:
            with open(config_file) as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            return Respond(request, 'config_corrupt', "text/plain", 200)

        if not 'default' in data:
            return Respond(request, 'config_incorrect', "text/plain", 200)
        elif not 'password' in data['default'] or not 'registered' in data['default']:
            return Respond(request, 'config_incorrect', "text/plain", 200)

        password = data['default']['password']
        password = str.encode(password)

        # Global password decription
        key = 'vei8dcIyZU-XeMiydKm6DVRJSjhjgQ_bA0h6pqHD1Q4='
        f = Fernet(key)
        decrypted_password = f.decrypt(password)
        decrypted_global_password = decrypted_password.decode("utf-8")

        # Query parameters
        name = request.args.get('name')
        password = request.args.get('pw')
        glob_password = request.args.get('globpw')
        folder_name = request.args.get('folder')

        # REGISTER VALIDATION
        # username's length is more then 6
        if (len(name) < 6):
            return Respond(request, 'username_length', "text/plain", 200)
        # password's length is more then 6
        if (len(password) < 6):
            return Respond(request, 'pw_length', "text/plain", 200)
        # password contains uppercase letter
        if (not any(x.isupper() for x in password)):
            return Respond(request, 'pw_upper', "text/plain", 200)
        # password contains lowercase letter
        if (not any(x.islower() for x in password)):
            return Respond(request, 'pw_lower', "text/plain", 200)
        # password contains digit
        if (not any(x.isdigit() for x in password)):
            return Respond(request, 'pw_digit', "text/plain", 200)
        # Global password incorrect
        if glob_password != decrypted_global_password:
            return Respond(request, 'globalpw', "text/plain", 200)

        # Local password encription
        encoded_password = password.encode()
        encrypted_password = f.encrypt(encoded_password)
        decoded_password = encrypted_password.decode("utf-8")

        registered_folders = []
        if 'registered' in data['default']:
            for i in data['default']['registered']:
                registered_folders.append(i['foldername'])

        if ('registered' in data['default']):
            # Check the username or sublogdir is taken (1 user - 1 sublogdir)
            for i in data['default']['registered']:
                if i['name'] == name:
                    return Respond(request, 'taken_username', "text/plain", 200)
                if i['foldername'] == folder_name:
                    return Respond(request, 'taken_folder', "text/plain", 200)

            data['default']['registered'].append(
                {'name': name, 'password': decoded_password, 'foldername': folder_name})
        else:
            data['default']['registered'] = [{'name': name, 'password': decoded_password, 'foldername': folder_name}]

        with open(os.path.join(self.logdir, 'cnf.json'), 'w') as fp:
            json.dump(data, fp)

        # Dont move the files if the sublogdir, plugins.json and cnf.json are exist
        if os.path.isdir(os.path.join(self.logdir, folder_name)):
            if os.path.isfile(os.path.join(self.logdir, folder_name, 'plugins.json')) and os.path.isfile(
                    os.path.join(self.logdir, folder_name, 'cnf.json')):
                return Respond(request, 'okwo', "text/plain", 200)

        # Create the sublogdir and move the main logdir files to the sublogdir
        files = glob.glob(self.logdir + '/*')
        file_names = []

        for i in files:
            file_names.append(i.split('/')[-1])

        # create sublogdir
        os.mkdir(self.logdir + '/' + folder_name)

        for i in registered_folders:
            if i in file_names:
                file_names.remove(i)

        # move files to sublogdir
        for i in file_names:
            if '.' in i:
                os.popen('cp ' + self.logdir + '/' + i + ' ' + self.logdir + '/' + folder_name + '/' + i)
            else:
                os.popen('cp -r ' + self.logdir + '/' + i + '/' + ' ' + self.logdir + '/' + folder_name + '/' + i + '/')

        return Respond(request, 'ok', "text/plain", 200)

    def _encrypt_string(self, string: str) -> str:
        """
        Encrypts a string using Fernet encryption.

        Args:
            string: A string to be encrypted.

        Returns:
            A string representing the encrypted version of the input string.
        """

        try:
            encoded = string.encode()
            encrypted = self.f.encrypt(encoded)
            decoded = encrypted.decode("utf-8")
            return decoded
        except Exception as e:
            raise ValueError(f"Unable to encrypt string: {e}")

    @wrappers.Request.application
    def _serve_encrypt(self, request):
        """
        Encrypts the logdir and the disabled/enabled features.

        Args:
            request: A Flask request object.

        Returns:
            A tuple containing a string representing the encrypted version of the input string, a string representing
            the response type, and an integer representing the HTTP status code.
        """

        try:
            string = request.args.get('string')
            encrypted = self._encrypt_string(string)
            return Respond(request, encrypted, "text/plain", 200)
        except Exception as e:
            raise ValueError(f"Unable to serve encrypt request: {e}")

    def _decrypt_string(self, string: str) -> str:
        """
        Decrypts an encrypted string using Fernet encryption.

        Args:
            string: A string to be decrypted.

        Returns:
            A string representing the decrypted version of the input string.
        """

        try:
            encoded = string.encode()
            decrypted = self.f.decrypt(encoded)
            decoded = decrypted.decode("utf-8")
            return decoded
        except Exception as e:
            raise ValueError(f"Unable to decrypt string: {e}")

    @wrappers.Request.application
    def _serve_decrypt(self, request):
        """
        Decrypts the encrypted URL.

        Args:
            request: A Flask request object.

        Returns:
            A tuple containing a string representing the decrypted version of the input string, a string representing
            the response type, and an integer representing the HTTP status code.
        """

        try:
            string = request.args.get('string')
            decrypted = self._decrypt_string(string)
            return Respond(request, decrypted, "text/plain", 200)
        except Exception as e:
            raise ValueError(f"Unable to serve decrypt request: {e}")

    @wrappers.Request.application
    def _serve_boardpath(self, request):
        """
            Returns the absolute path to the board.

            Args:
                request: A Flask request object.

            Returns:
                A tuple containing a string representing the absolute path to the board, a string representing the response
                type, and an integer representing the HTTP status code.
            """

        try:
            path = os.path.dirname(os.path.realpath(__file__)).split("/")[:-3]
            path = '/' + os.path.join(*path)
            print("Board path: ", path)
            return Respond(request, path, "text/plain", 200)
        except Exception as e:
            raise ValueError(f"Unable to serve boardpath request: {e}")

    @wrappers.Request.application
    def _serve_testexit(self, request):
        """
        Triggers the exit of the program.

        Args:
            request: A Flask request object.

        Returns:
            A tuple containing a string representing the message "exit", a string representing the response type, and an
            integer representing the HTTP status code.
        """

        try:
            program._trigger_exit()
            return Respond(request, "exit", "text/plain", 200)
        except Exception as e:
            raise ValueError(f"Unable to serve testexit request: {e}")

    @wrappers.Request.application
    def _serve_forgotpassword(self, request):
        """
        Handles the password reset functionality.

        Args:
            request: A Flask request object.

        Returns:
            A tuple containing a string representing the response message, a string representing the response type, and an
            integer representing the HTTP status code.
        """

        try:
            name = request.args.get('name')
            globalpw = request.args.get('global')
            password = request.args.get('password')
            conform_password = request.args.get('passwordconfirm')

            if password != conform_password:
                return Respond(request, "Password and Confirm password don't match!", "text/plain", 500)

            # Config file
            config_file = os.path.join(self.logdir, 'cnf.json')
            with open(config_file) as json_file:
                data = json.load(json_file)

            password_correct = data['default']['password'].encode()

            # Global password decryption
            key = 'vei8dcIyZU-XeMiydKm6DVRJSjhjgQ_bA0h6pqHD1Q4='
            f = Fernet(key)
            decrypted_password = f.decrypt(password_correct)
            decrypted_global_password = decrypted_password.decode("utf-8")

            if decrypted_global_password != globalpw:
                return Respond(request, "Invalid global password!", "text/plain", 500)

            # Encrypt new password
            encoded = password.encode()
            encrypted = f.encrypt(encoded)
            decoded = encrypted.decode("utf-8")

            user = False
            if 'registered' in data['default']:
                # Check if the username or sublogdir is taken (1 user - 1 sublogdir)
                for i in data['default']['registered']:
                    if i['name'] == name:
                        i['password'] = decoded
                        user = True

            with open(os.path.join(self.logdir, 'cnf.json'), 'w') as fp:
                json.dump(data, fp)

            if not user:
                return Respond(request, "Username not found!", "text/plain", 500)

            return Respond(request, "ok", "text/plain", 200)

        except Exception as e:
            raise ValueError(f"Unable to serve forgotpassword request: {e}")
