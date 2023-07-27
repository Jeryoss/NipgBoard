import argparse
import os
import json
from cryptography.fernet import Fernet

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder",
        type=str,
        help="The directory to fill as an NIPGBoard logdir with configuration files.",
        default="",
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        help="The administrative password users need to register new accounts with personal passwords",
        default="adminpassword",
    )
    parser.add_argument(
        "-im", "--images",
        nargs="+",
        type=str,
        help="Image directories to be added into the configuration, use -im or --help.",
        default=[],
    )
    args = parser.parse_args()

    logdir = args.folder
    passw = args.password
    images = args.images
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    key = 'vei8dcIyZU-XeMiydKm6DVRJSjhjgQ_bA0h6pqHD1Q4='
    encoded_admin_password = passw.encode()
    f = Fernet(key)
    encrypted_admin_password = f.encrypt(encoded_admin_password)

    cwd = os.getcwd()
    algs = os.path.join(cwd, 'algorithms')
    cnf = {
        "default": {
            "password": "",
            "admin_password": encrypted_admin_password.decode('UTF-8'),
            "embedding_folder": "embedding",   #TODO: remove these two
            "name": "embedding",
            "image_folders": images,
            "video_folder": "video",
            "registered": [],
        },
        "trainings": [
            {"algorithm_path": algs, "embedding_folder": "denrun_paired", "algorithm": {"callable": "run", "arguments": [], "file": "kira", "keyword_arguments": {"epoch": "10"}}, "name": "DenseNet Paired", "type": "train"},
            {"algorithm_path": algs, "embedding_folder": "denrun", "algorithm": {"callable": "run", "arguments": [], "file": "densenet_base", "keyword_arguments": {"epoch": "10"}}, "name": "DenseNet", "type": "base"},
            {"algorithm_path": algs, "embedding_folder": "denrun_organized", "algorithm": {"callable": "run", "arguments": [], "file": "kira_improve", "keyword_arguments": {"epoch": "10"}}, "name": "DenseNet Organized", "type": "train"},
        ]
    }
    plugs = {
        "plugins": ["modelmanager", "selected", "image", "projector", "executer", "graphcut"]
    }

    with open(os.path.join(logdir, 'cnf.json'), 'w') as fp:
        json.dump(cnf, fp)
    
    with open(os.path.join(logdir, 'plugins.json'), 'w') as fp:
        json.dump(plugs, fp)
    