from datetime import datetime
import json
import os
import re


def datetime_now_string():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def tmp_dir(filepath):
    tmp_path = filepath + "tmp/"
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    return tmp_path


def get_files(path, regex):
    """Retrieves all data files that match the ``regex`` in the
    directory specified by ``path``.
    """
    return [
        os.path.join(path, f)
        for f in os.listdir(path)
        if (
            f.endswith(".json")
            and os.path.isfile(os.path.join(path, f))
            and bool(re.match(regex, f))
        )
    ]


def write_json(json_dict, filename):
    """Writes the dictionary to JSON file with name ``filename``.
    :param json_dict: The dictionary to write as a JSON file.
    :type json_dict: dict
    :param filename: The name of the JSON file. Note that ``.json`` extension is automatically added.
    :type filename: string
    :returns: ``None``
    """

    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_dict, indent=2))


def read_json(filename):
    """Reads data from a JSON file.
    :param filename: The path to the JSON file. Note this string must contain the ``.json`` extension.
    :type filename: string
    :returns: The dictionary read from the JSON file.
    :rtype: dict
    """

    with open(filename) as file:
        opt_dict = json.load(file)

    return opt_dict
