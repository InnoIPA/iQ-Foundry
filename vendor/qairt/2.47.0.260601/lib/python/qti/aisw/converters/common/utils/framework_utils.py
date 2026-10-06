# ==============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import multiprocessing
import queue
import signal
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Text, Tuple, Union

import sys
try:
    import rich
    from rich.console import Console
    from io import StringIO
except:
    pass

import numpy as np
from qti.aisw.converters.common.converter_ir.axis_tracker import AxisTracker
from qti.aisw.converters.common.utils.converter_utils import log_debug1, log_warning


class TensorInfo:
    def __init__(self, name: str, shape: List, dtype: Text, layout) -> None:
        """
        Store the properties of the tensor.

        :param str name: Name of the tensor
        :param List shape: shape of the tensor
        :param str type: type of the tensor.
        :param layout: layout of the tensor.
        """
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def __check_shapes(self, shape1: List, shape2: List) -> bool:
        """
        Check different shapes values of given tensors, including the dynamic tensor shapes.

        :param List shape1: Shape of tensor-1
        :param List shape2: Shape of tensor-2
        :return bool: True if the shapes are matching else False.
        """
        if len(shape1) != len(shape2):
            return False
        if shape1 == shape2:
            return True

        for s1, s2 in zip(shape1, shape2):
            if (isinstance(s1, int) and isinstance(s2, int)) and (s1 != s2):
                if (s1 == -1 and s2 > 0):
                    return True
                if (s1 > 0 and s2 == -1):
                    return True
                return False
        return True

    def __eq__(self, other) -> bool:
        """
        Compare the current TensorInfo object with the user provided object.

        :param TensorInfo other: TensorInfo object to compare with.
        :return bool: True if both are same else False.
        """
        if (
            (self.name == other.name)
            and (self.__check_shapes(self.shape, other.shape))
            and (self.dtype == other.dtype)
            # No need to check layout because for dynamic shaped tensors the
            # layout can be NONTRIVIAL
            # and (self.layout == other.layout)
        ):
            return True
        return False


class FrameworkSummary:

    """
    This is common data structure which will we used in various framework to
    provide the summary of model. e.g. input/output info, number of ops and
    their count etc.
    """

    def __init__(
        self,
        ir_version: Optional[Text] = "",
        producer_name: Optional[Text] = "",
        ops_counter: Optional[Dict] = None,
        total_parameters: Optional[Text] = "",
        inp_specs: Optional[Dict[Text, TensorInfo]] = None,
        out_specs: Optional[Dict[Text, TensorInfo]] = None,
        model_name: Optional[Text] = "",
    ):
        """
        Base Summarization Class

        :param Optional[Text] ir_version: Version of IR, defaults to ""
        :param Optional[Text] producer_name: Producer name, defaults to ""
        :param Optional[Dict] ops_counter: Counts of each operation in the model, defaults to None
        :param Optional[Text] total_parameters: total number of per, defaults to ""
        :param Optional[Dict[Text, TensorInfo]] inp_specs:Input tensor information, defaults to None
        :param Optional[Dict[Text, TensorInfo]] out_specs: output tensor information, defaults to None
        :param Optional[Text] model_name: Name of the model, defaults to ""
        """
        self.ir_version = ir_version
        self.producer_name = producer_name
        self.ops_counter = ops_counter
        self.total_parameters = total_parameters
        self.inp_specs = inp_specs
        self.out_specs = out_specs
        self.model_name = model_name

    def __repr__(self) -> str:
        """
        Get the string representation of the Summary object.
        """
        if "rich" in sys.modules:
            # print the model summary using rich library if available
            summary_str = ""
            summary_str += f"IR Version: {self.ir_version}\n" if self.ir_version != "" else ""
            summary_str += f"Total Count of Ops: {sum(self.ops_counter.values())}\n"
            summary_str += f"Total Model Parameters: {int(self.total_parameters):,}\n"
            summary_str += "All Ops: " + str(dict(self.ops_counter)).replace("{", "{{").replace("}", "}}")

            console = Console(record=True, file=StringIO())
            p = rich.panel.Panel(summary_str, title=f"{self.model_name} Summary", highlight=True)
            console.print(p, width=120, new_line_start=True)

            table = rich.table.Table(
                title=f"{self.model_name} Input-Output Details",
                caption="Table Generated by QNN Converter Tool",
                expand=True,
                row_styles=["none", "none"],
                pad_edge=True,
                box=rich.box.ROUNDED,
            )
            table.add_column("[yellow]Name")
            table.add_column("[cyan]Shape")
            table.add_column("[blue]Input/Output")
            table.add_column("[green]Dtype")
            for k, v in self.inp_specs.items():
                table.add_row(k, str(v[0]), v[1], v[2])
            for k, v in self.out_specs.items():
                table.add_row(k, str(v[0]), v[1], v[2])
            console.print(table, width=120, new_line_start=True)

            data = console.export_text(styles=False)
        else:
            log_warning("rich library is not found. Skipping pretty print for model summary")
            # Print the model summary without pretty print
            def format_ops_string(ops_count, limit=90):
                """ Build and format Ops string with count """
                ops_string = "All Ops:\n"
                tmp = ""
                for op, count in ops_count.items():
                    op_str = op + ": " + str(count)
                    # Go to new line if the Ops string exceeds character limit
                    if len(tmp) + len(op_str) > limit:
                        ops_string += "\t" + tmp + "\n"
                        tmp = ""
                    tmp += op_str + ", "
                # Add last line to Ops string and strip comma at the end
                ops_string += "\t" + tmp.rstrip(", ") + "\n"
                return ops_string

            summary_str = "\n" + "=" * 100 + "\n"
            summary_str += self.model_name + " Summary\n"
            summary_str += "=" * 100 + "\n"
            summary_str += f"IR Version: {self.ir_version}\n" if self.ir_version != "" else ""
            summary_str += f"Total Count of Ops: {sum(self.ops_counter.values())}\n"
            summary_str += f"Total Model Parameters: {int(self.total_parameters):,}\n"
            summary_str += format_ops_string(self.ops_counter)
            summary_str += "Inputs:\n"
            for k, v in self.inp_specs.items():
                summary_str += "\t" + k + " (shape:" + str(v[0]) + ", datatype:" + v[2] + ")\n"
            summary_str += "Outputs:\n"
            for k, v in self.out_specs.items():
                summary_str += "\t" + k + " (shape:" + str(v[0]) + ", datatype:" + v[2] + ")\n"
            summary_str += "=" * 100 + "\n"
            data = summary_str

        return data


def generate_test_data(
    input_info_dict: Dict[Text, TensorInfo]
) -> Dict[str, np.ndarray]:
    """
    Generate the test inputs based on given shape and data type
    :param:input_info_dict (Dict[str, Dict]): A dict with mapping from input name to
    another dict having info regarding input shape and input dtype.

    :returns: Dict[str, np.ndarray]: A dict with mapping from input name to test data
    of the input in np.array format.
    """
    final_inputs = OrderedDict()
    for input_name, tensor in input_info_dict.items():
        input_shape = [s if s != -1 else 1 for s in tensor.shape]
        input_dtype = tensor.dtype
        final_inputs[input_name] = np.ones(input_shape).astype(input_dtype)
    return final_inputs


def determine_layout(tensor_shape: Union[Tuple[int], List[int]]) -> Text:
    """
    Gets the tensor shape layout of a given tensor.

    :param :tensor_shape (Tuple[int]): The shape, including batch dimension.
    :returns: Text: The determined data layout.
    """
    layout = AxisTracker.AxisFormat
    if not isinstance(tensor_shape, Tuple) and not isinstance(tensor_shape, List):
        log_debug1(f"Tensor Layout cant be obtained from tensor shape: {tensor_shape}")
        return layout.NONTRIVIAL

    # The ratio is closer to ~1, the closer a and b are.
    def minmax_ratio(a, b):
        return abs(max(a, b) / min(a, b))

    # Assume all shapes includes unchanged batch dimension at index 0, so we
    # need to check shape[1:].
    if len(tensor_shape) == 4:
        unknown_cnt = sum(
            [1 if not isinstance(s, int) else 0 for s in tensor_shape[1:]]
        )
        if unknown_cnt >= 1:
            log_debug1(
                f"One or more unknown shape value present in tensor shape: {tensor_shape}."
            )
            if 3 in tensor_shape and unknown_cnt == 2:
                # This means the shape is [unk, 3, unk, unk] or [unk, unk, unk, 3]
                # unknown_cnt == 2 means two axes other than batch dim are dynamic.
                # If we found 3 channels then we can determine layout based
                # the index at which 3 channels occur.
                idx = tensor_shape.index(3)
                if idx == 1:
                    return layout.NCS
                elif idx == 3:
                    return layout.NSC
            else:
                return layout.NONTRIVIAL
        # Typically, H and W are quite close,
        # so if minmax_ratio(0, 1) > minmax_ratio(1, 2), then we assume CHW.
        if minmax_ratio(tensor_shape[1], tensor_shape[2]) > minmax_ratio(
            tensor_shape[2], tensor_shape[3]
        ):
            return layout.NCS
        return layout.NSC
    elif len(tensor_shape) == 5:
        unknown_cnt = sum(
            [1 if not isinstance(s, int) else 0 for s in tensor_shape[1:]]
        )
        if unknown_cnt >= 1:
            log_debug1(
                f"One or more unknown shape value present in tensor shape: {tensor_shape}."
            )
            return layout.NONTRIVIAL
        # For yolo models. Also need to check this for
        # spatio temporal models e.g. 3DCNN models
        if 1 in tensor_shape[1:]:
            # For YoloX with No anchors as it is anchor less detector.
            anchor_idx = list.index(list(tensor_shape[1:]), 1) + 1
        elif 3 in tensor_shape[1:]:
            # For Yolos with 3 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 3) + 1
        elif 5 in tensor_shape[1:]:
            # For YoloV2 with 5 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 5) + 1
        elif 9 in tensor_shape[1:]:
            # For YoloV2 with 9 anchors
            anchor_idx = list.index(list(tensor_shape[1:]), 9) + 1
        else:
            return layout.NONTRIVIAL

        if anchor_idx == 1:
            return layout.NDHWC
        elif anchor_idx == 2:
            return layout.NCDHW
        else:
            return layout.NONTRIVIAL

    elif len(tensor_shape) == 3:
        return layout.NFC

    elif len(tensor_shape) == 2:
        return layout.NC
    else:
        log_debug1(f"Cannot determine layout for tensor with shape: {tensor_shape}.")
        return layout.NONTRIVIAL


class SpawnFuncWrapper:
    """
    Function wrapper used for executing underlying functions and add function results to
    shared queue.
    """

    # This function wrapper should be in module namespace, as this hard requirement is coming from
    # WINDOWS OS. Don't move this class inside spawn_process_and_exec that way it will become part of
    # local namespace.
    def __init__(self, func):
        """
        Initializing the function wrapper.

        :param func: Function to wrap.
        """
        self.func = func

    def __call__(self, process_name, shared_queue, *args, **kwargs):
        """
        Execute the underlying function and add results to shared queue.

        :param  shared_queue: Add results to shared queue system.
        """
        process = multiprocessing.current_process()
        log_debug1(f"Executing sub-process: {process_name}, Process ID: {process.pid}")
        try:
            res = self.func(*args, **kwargs)
            shared_queue.put(res)
        except Exception as e:
            log_warning(f"Process Failed with exception : {e}")


def spawn_process_and_exec(func, *args, **kwargs):
    """
    Execute the function in spawned process.

    :param func: Function to execute.
    :return : Results from the function.
    """
    # Note: For linux os, multiprocess by default uses "fork" process, whereas
    #       for windows and macos it is "spawn" process by default. Using
    #       spawn explicitly as it creates new interpreter and imports required
    #       modules rather than replicating parent memory blocks.
    process_name = kwargs.pop("process_name", "Process")
    res = None
    status = False
    try:
        with multiprocessing.get_context("spawn").Manager() as manager:
            shared_queue = manager.Queue()
            args = process_name, shared_queue, *args
            process = multiprocessing.Process(
                target=SpawnFuncWrapper(func), args=args, kwargs=kwargs
            )
            process.start()
            process.join()
            try:
                res = shared_queue.get(block=False)
            except queue.Empty as e:
                if process.exitcode == -signal.SIGSEGV:
                    log_warning(f"Segmentation fault occur when running {process_name}")
                    # The child process was terminated by a certain signal during execution.
                elif process.exitcode != 0:
                    log_warning(f"Unexpected error occur when running {process_name}")
            status = res is not None
    except Exception as e:
        log_warning(
            f"While executing {process_name} process, it failed with " \
            f"exception : {e}"
        )

    return status, res


def make_dirs(filename: Text) -> Text:
    """
    create directory recursively.

    :param Text filename: Filename to create directory.
    :return Text: filename.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return filename
