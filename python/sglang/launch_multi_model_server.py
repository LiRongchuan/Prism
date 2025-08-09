"""Launch the inference server."""

import os
import sys

from sglang.multi_model.multi_model_server import launch_multi_model_server
from sglang.multi_model.multi_model_server_args import prepare_server_args
from sglang.srt.utils import kill_child_process

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_multi_model_server(server_args)
    except Exception as e:
        raise e
    finally:
        kill_child_process(os.getpid(), including_parent=False)
