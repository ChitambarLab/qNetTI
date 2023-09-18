from datetime import datetime
import json
import os
import re


def datetime_now_string():
    """
    Retrieves the current datetime in universal time coordinates (UTC).

    :returns: The current datetime formatted as ``YYYY-MM-DDTHH-mm-SSZ``
    :rtype: String
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


def tmp_dir(filepath):
    """
    Creates a temporary directory ``tmp/``  in the directory
    specified by ``filepath``.

    :param filepath: The filepath to the directory in
                     which to create a temporary directory.
    :type filepath: String

    :returns: The path to ``"filepath/tmp/"``
    :rtype: String
    """
    tmp_path = filepath + "tmp/"
    if not os.path.isdir(tmp_path):
        os.mkdir(tmp_path)

    return tmp_path


def mkdir(filepath, dir_name):
    """
    Creates a directory ``dir_name/``  in the directory
    specified by ``filepath``.

    :param filepath: The filepath to the directory in
                     which to create the directory.
    :type filepath: String

    :param dir_name: The name of the directory to create.
    :type dir_name: String

    :returns: The path to ``"filepath/dir_name/"``
    :rtype: String
    """
    dir_path = filepath + dir_name
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    return dir_path


def get_files(path, regex):
    """Retrieves all data files that match the ``regex`` in the
    directory specified by ``path``.

    :param path: A file path to search in.
    :type path: string

    :param regex: A regular expression.
    :type regex: regex

    :returns: A list of file names that match the specified regex.
    :rtype: list[String]
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
