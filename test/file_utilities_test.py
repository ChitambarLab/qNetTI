import pytest
import re
import os
import shutil

import qnetti


def test_datetime_now_string():
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$", qnetti.datetime_now_string())


@pytest.fixture()
def tmp_dir_cleanup():
    tmp_dir = "./test/tmp/"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)

    assert not os.path.exists(tmp_dir)

    yield

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    assert not os.path.exists(tmp_dir)


def test_tmp_dir(tmp_dir_cleanup):
    assert not os.path.exists("./test/tmp/")

    tmp_dir = qnetti.tmp_dir("./test/")

    assert os.path.exists("./test/tmp/")
    assert tmp_dir == "./test/tmp/"


def test_get_files(tmp_dir_cleanup):
    tmp_dir = qnetti.tmp_dir("./test/")

    assert qnetti.get_files(tmp_dir, r".*") == []

    qnetti.write_json({"elephant": 123}, tmp_dir + "test_get_files_elephant")
    qnetti.write_json({"blah": "hello"}, tmp_dir + "test_get_files_blah")

    filenames = qnetti.get_files(tmp_dir, r"test_get_files.*")
    assert len(filenames) == 2

    filenames = qnetti.get_files(tmp_dir, r"test_get_files_elephant")
    assert len(filenames) == 1

    filenames = qnetti.get_files(tmp_dir, r"test_get_files_blah")
    assert len(filenames) == 1

    assert qnetti.get_files(tmp_dir, r"no test file matches this string") == []


@pytest.mark.parametrize(
    "json_dict",
    [
        {
            "cost_vals": [0.0, 1.0, 1.2],
            "settings_history": [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]],
            "step_size": 0.1,
            "datetime": "2023-01-31T12-12-12Z",
            "num_steps": 2,
            "optimizer": "GradientDescentOptimizer",
        },
        {"test_key": "val", "other_key": 123456789},
    ],
)
def test_read_write_json(tmp_dir_cleanup, json_dict):
    tmp_dir = qnetti.tmp_dir("./test/")
    filename = "test_read_write_optimization_json"

    qnetti.write_json(json_dict, tmp_dir + filename)

    assert os.path.isfile(tmp_dir + filename + ".json")

    read_json_dict = qnetti.read_json(tmp_dir + filename + ".json")

    assert read_json_dict == json_dict

