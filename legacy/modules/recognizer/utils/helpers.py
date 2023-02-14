import os
import shutil
from os import environ as env
from subprocess import run


def loadw(sink, weight_path=None, fallback_url=None, *args, **kwargs):
    if weight_path is None:
        weight_path = env.get("MODEL_WEIGHT_PATH")

    if fallback_url is None:
        fallback_url = env.get("MODEL_WEIGHT_URL") or env.get(
            "MODEL_WEIGHT_PATH")

    debug_info = """
    DEBUG INFO:
        weight_path: {weight_path}
        fallback_url: {fallback_url}
    """

    if os.path.isdir(weight_path):
        raise Exception(f"{weight_path} is a directory\n{debug_info}")

    if not os.path.exists(weight_path):
        if shutil.which("wget") is None:
            raise Exception("Install wget first\n{debug_info}")
        if fallback_url is None:
            raise Exception(
                "Model weight {weight_path} not found, specify, {fallback_url}\n{debug_info}"
            )
        dirname = os.path.dirname(weight_path)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True, mode=755)
        run(f"wget {fallback_url} -O {weight_path}", shell=True)

    return sink(weight_path, *args, **kwargs)
