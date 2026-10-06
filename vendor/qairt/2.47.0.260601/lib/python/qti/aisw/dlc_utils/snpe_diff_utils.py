#!/usr/bin/env python3
# -*- mode: python -*-
#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================
from __future__ import print_function
import argparse
import logging
import os
import sys
import csv
from collections import OrderedDict, defaultdict
import numpy as np

from qti.aisw.converters.common.utils.converter_utils import log_error

try:
    from qti.aisw.dlc_utils import snpe_dlc_utils
except ImportError as ie:
    print("Failed to find necessary package:")
    print(str(ie))
    print("Please ensure that $SNPE_ROOT/lib/python is in your PYTHONPATH")
    sys.exit(1)

# Assign commonly used functions from snpe.dlc_utils.py to variables
print_value = snpe_dlc_utils.print_value
print_row = snpe_dlc_utils.print_row
get_si_notation = snpe_dlc_utils.get_si_notation

graphName1 = ""
graphName2 = ""

copyrights_str = "--copyrights"
layers_str = "--layers"
parameters_str = "--parameters"
dimensions_str = "--dimensions"
weights_str = "--weights"
outputs_str = "--outputs"
output_tensor_types_str = "--compare_output_tensor_types"
diff_by_id_str = "--diff_by_id"
hta_str = "--hta"

# Helpers
class DiffStrategies(object):
    """
    Contains supported model layer comparision strategies
    """
    # compares only layers that are same names between models
    BY_COMMON_NAMES = "by_common_names"
    # compares all layers traversing by id and type
    # Note: comparision will halt if more than one adjacent layer type doesnt match
    BY_LAYER_IDS = "by_layer_ids"

    diff_strategies = [BY_COMMON_NAMES, BY_LAYER_IDS]


def get_shared_layers(m1_dict, m2_dict):
    """ gets the layers with same names in both dictionaries. Note: requires keys to be name of layer
    @param m1_dict: dictionary of arbitrary value for first DLC model
    @param m2_dict: dictionary of arbitrary value for second DLC model

    @return a dict containing common layers between 2 dlcs
    """

    m1_layers = OrderedDict.fromkeys(list(m1_dict.keys()))
    m2_layers = OrderedDict.fromkeys(list(m2_dict.keys()))
    shared_layers = list(OrderedDict.fromkeys(x for x in m1_layers if x in m2_layers))

    return shared_layers


def check_layer_types_and_orders_comparable(m1, m2):
    """ checks if 2 models have 2 consecutive layer types that are out of order
    @param m1: first DLC model
    @param m2: second DLC model

    @return True if models do not have 2 consecutive layer types that are out of order along with the layer ids,
            False otherwise
    """

    # Dictionary of layer id and type
    m1_types_info = m1.types_info_by_id()
    m2_types_info = m2.types_info_by_id()

    # sort by id and verify order
    m1_sorted_info = list(sorted(m1_types_info.items(), key=lambda id:(id[0])))
    m2_sorted_info = list(sorted(m2_types_info.items(), key=lambda id:(id[0])))
    smaller_model_info = m1_sorted_info if len(m1_sorted_info) < len(m2_sorted_info) else m2_sorted_info

    out_of_order_count = 0
    out_of_order_ids = []
    for layer_id, _ in smaller_model_info:
        if m1_sorted_info[layer_id] != m2_sorted_info[layer_id]:
            out_of_order_count += 1
            out_of_order_ids.append(layer_id)
        else:
            out_of_order_count = 0  # reset to only account for consecutive layer types not being the same
            out_of_order_ids = []

        # no need to continue models are not comparable to diff sequentially
        if out_of_order_count > 1:
            return False, out_of_order_ids

    return True, out_of_order_ids


def is_empty(val):
    """ Simple check to see if value is empty"""
    if type(val) is np.ndarray:
        return val.size == 0
    return val is None or val == [] or val == ""


def get_layers_with_value_diffs(m1, m2, comp_func, diff_var_str, diff_strategy=DiffStrategies.BY_COMMON_NAMES):
    """ gets the layers with same names in both dictionaries. Note: requires keys to be name of layer
    @param m1: first DLC model
    @param m2: second DLC model
    @param comp_func: function to call to retrieve value for use_case. E.g snpe_dlc_utils.params_info
    @param diff_var_str: a string representing what is being compared. This will be used to give users an
                         informative message.
    @param diff_strategy: optional variable used to determine how models should be compared. See DiffStrategies class
                           to see different options and their descriptions.

    @return: A list containing a boolean variable, a list, and a string.

            Boolean:    True:   No value differences between shared layers
                        None:   No shared layers, thus parameters not comparable
                        False:  Value differences exist between certain shared layers

            List:       Contains two dictionaries
                        - Model 1 layers with value differences
                        - Model 2 layers with value differences

            String:     Info message for user
    """

    def _update_diff(m1_name_, m2_name_, m1_id_, m2_id_):
        m1_value, m2_value = m1_info.get(m1_name_), m2_info.get(m2_name_)
        # Only make comparison if either one model has a value. For instance: weight values are not applicable
        # to all layers
        if not is_empty(m1_value) or not is_empty(m2_value):
            if type(m1_value) is list and type(m2_value) is list:
                if len(m1_value) != len(m2_value):
                    value_diff = True
                else:
                    # Handle the case where we have lists of nd_arrays to compare
                    def weights_are_same(v1, v2):
                        if v1.shape != v2.shape or v1.dtype != v2.dtype:
                            return False
                        return np.all(v1 == v2) or np.allclose(v1, v2, equal_nan=True)

                    if all([all(type(z) is np.ndarray for z in zz) for zz in [m1_value, m2_value]   ]):
                        for v1, v2 in zip(m1_value, m2_value):
                            value_diff = not weights_are_same(v1, v2)

                    else:
                        value_diff = m1_value != m2_value

            elif type(m1_value != m2_value) is np.ndarray:
                # Check if any value in arrays are different. Need shape check first since numpy.allclose
                # broadcasts if shape not equal
                value_diff = m1_value.shape != m2_value.shape or \
                             not np.allclose(m1_value, m2_value, equal_nan=True)
            else:
                value_diff = m1_value != m2_value
            if value_diff:
                m1_diff_values.update({m1_name_: (m1_value, m1_id_)})
                m2_diff_values.update({m2_name_: (m2_value, m2_id_)})

    if diff_strategy not in DiffStrategies.diff_strategies:
        log_error("Unknown DiffStrategy: {}. Please choose from: {} ", str(diff_strategy),
                  str(DiffStrategies.diff_strategies))

    # Dictionary of layer name and info
    m1_info = comp_func(m1, graphName1)
    m2_info = comp_func(m2, graphName2)
    if diff_var_str == 'Weight':
        _, _, _, _, _, _, is_stripped_1 = snpe_dlc_utils.ModelInfo.extract_model_info(m1, graphName1)
        _, _, _, _, _, _, is_stripped_2 = snpe_dlc_utils.ModelInfo.extract_model_info(m2, graphName2)
        if is_stripped_1 and is_stripped_2:
            sys.exit("Model 1 and Model 2 both have no weights, not able to compare weights with each other.")
        if is_stripped_1:
            sys.exit("Model 1 has no weights, not able to compare weights with other models.")
        if is_stripped_2:
            sys.exit("Model 2 has no weights, not able to compare weights with other models.")

    # Dictionary of layer name and id info
    m1_id_layer = m1.ids_layer()
    m2_id_layer = m2.ids_layer()

    m1_num_layers = len(m1_id_layer.items())
    m2_num_layers = len(m2_id_layer.items())

    m1_diff_values = OrderedDict()
    m2_diff_values = OrderedDict()
    shared_layers_len = 0
    diff_info_msg = ""
    if diff_strategy == DiffStrategies.BY_LAYER_IDS:
        are_comparable, layer_ids = check_layer_types_and_orders_comparable(m1, m2)
        if are_comparable:
            diff_info_msg = '[Note: Comparison done by ordering layers using ids and diffing across.]'
            m1_sorted = list(sorted(m1_id_layer.items(), key=lambda id:(id[1])))
            m2_sorted = list(sorted(m2_id_layer.items(), key=lambda id:(id[1])))
            shared_layers_len = len(m1_sorted) if len(m1_sorted) < len(m2_sorted) else len(m2_sorted)
            for i in range(shared_layers_len):
                m1_name, m1_id = m1_sorted[i]
                m2_name, m2_id = m2_sorted[i]
                _update_diff(m1_name, m2_name, m1_id, m2_id)

            # Add remaining layers unique to the larger model that have values
            if len(m1_sorted) != shared_layers_len:
                for j in range(shared_layers_len, len(m1_sorted)):
                    m1_name, m1_id = m1_sorted[j]
                    if not is_empty(m1_info.get(m1_name)):
                        m1_diff_values.update({m1_name: (m1_info.get(m1_name), m1_id)})
            elif len(m2_sorted) != shared_layers_len:
                for j in range(shared_layers_len, len(m2_sorted)):
                    m2_name, m2_id = m2_sorted[j]
                    if not is_empty(m2_info.get(m2_name)):
                        m2_diff_values.update({m2_name: (m2_info.get(m2_name), m2_id)})
        else:
            m1_names_info, m1_types_info = m1.op_names_by_id(graphName1), m1.types_info_by_id()
            m2_names_info, m2_types_info = m2.op_names_by_id(graphName2), m2.types_info_by_id()
            for id in layer_ids:
                m1_diff_values.update({m1_names_info.get(id): ([m1_types_info.get(id)], id)})
                m2_diff_values.update({m2_names_info.get(id): ([m2_types_info.get(id)], id)})
            return [None, [m1_diff_values, m2_diff_values], '\nWarning: Model1 and Model2 have 2 consecutive layers '
                                                            'with layer_type differences. ' + diff_var_str +
                                                            ' are not comparable ordered by id.\n', shared_layers_len]
    elif diff_strategy == DiffStrategies.BY_COMMON_NAMES:
        diff_info_msg = '[Note: Comparison done with identically named layers only.]'
        shared_layers = get_shared_layers(m1_info, m2_info)
        shared_layers_len = len(shared_layers)
        # add proposal to diff_by_id if either models have unique layer names(since those would have not been diffed)
        # and it is possible to compare by id
        by_id_comparable, _ = check_layer_types_and_orders_comparable(m1, m2)
        proposal = ""
        if (shared_layers_len != m1_num_layers or shared_layers_len != m2_num_layers) and by_id_comparable:
            proposal = "\nInfo: Try -i/" + diff_by_id_str + " for further analysis.\n"
            diff_info_msg += proposal  # add to info message to reuse recommendation for below as well.
        if shared_layers_len == 0:
            return [None, [m1_diff_values, m2_diff_values], '\nWarning: Model1 and Model2 have no identically named '
                                                            'layers. ' + diff_var_str + ' are not comparable.\n'
                                                            + proposal, shared_layers_len]

        for name in shared_layers:
            _update_diff(name, name, m1_id_layer.get(name), m2_id_layer.get(name))

    # Check if Model 1 and 2 have zero value differences
    if len(m1_diff_values) == 0 and len(m2_diff_values) == 0:
        return [True, [m1_diff_values, m2_diff_values], '\nInfo: Model1 and Model2 have no ' + diff_var_str +
                ' differences between layers. ' + diff_info_msg + '\n', shared_layers_len]
    else:
        return [False, [m1_diff_values, m2_diff_values], diff_info_msg, shared_layers_len]


def print_value_diffs_for_layers(m1_dict, m2_dict, input_dlc_one, input_dlc_two, num_shared_layers,
                                 csv_content, component_diffed, diff_strategy=DiffStrategies.BY_COMMON_NAMES):
    """
    Compares value differences between the two models' layers and outputs layers that exhibit differences
    in the values. An asterisks will be added as prefix to indicate which specific value is different

    @param m1_dict: dictionary of arbitrary value unique to first DLC model
    @param m2_dict: dictionary of arbitrary value unique to second DLC model
    @param input_dlc_one: First DLC file name
    @param input_dlc_two: Second DLC file name
    @param num_shared_layers: common layer names between models
    @param csv_content: output file to optionally save diff output to
    @param component_diffed: headers to use for table to display what is being diffed. Eg: Parameters
    @param diff_strategy: optional variable used to determine how models should be compared. See DiffStrategies class
                           to see different options and their descriptions.
    """
    def _print_diffs(ids_, m1_name, m2_name, m1_values_, m2_values_, is_val_arr_, col_sizes_, csv_content_):
        offset = len(prCyan("hello"))-len("hello")
        if is_val_arr_:
            # printing array values will be unreadable so only print layer ids/names
            if m1_name != m2_name:
                # show layer names from both models if they are different
                print_row([ids_, m1_name], col_sizes_, csv_content_)
                print_row(["", m2_name], col_sizes_, csv_content_)
            else:
                print_row([ids_, m1_name], col_sizes_, csv_content_)
        else:
            # loop through the first models values
            for m1_val in m1_values_:
                if not is_empty(m1_val):
                    if m1_val in m2_values_:
                        # print same values
                        print_row([ids_, m1_name, m1_val, m1_val], col_sizes_, csv_content_)
                        m2_values_.remove(m1_val)
                    else:
                        m1_val_key = m1_val.split(":")[0] + ":"
                        if m1_val_key in '\t'.join(m2_values_):
                            # print modified values
                            index_in_val2 = [m2_values_.index(m2_val) for m2_val in m2_values_ if m1_val_key in m2_val]
                            print_row_value_table([ids_, m1_name, '*' + m1_val, m2_values_[index_in_val2[0]]], col_sizes_, offset,
                                                  csv_content_)
                            m2_values_.remove(m2_values_[index_in_val2[0]])
                        else:
                            # print new values in model1
                            print_row_value_table([ids_, m1_name, '*' + m1_val, ''], col_sizes_, offset, csv_content_)

                        # clear name and ids since it only needs to get printed once
                        if m1_name != m2_name:
                            # show layer names from both models if they are different
                            print_row(["", m2_name, "", ""], col_sizes_, csv_content_)
                        m1_name = m2_name = ids_ = ''

            # print what is left in second model's value list
            for m2_val in m2_values_:
                if not is_empty(m2_val):
                    print_row_value_table([ids_, m2_name, '', '*' + m2_val], col_sizes_, offset, csv_content_)
                    # clear name and ids since it only needs to get printed once
                    m2_name = ids_ = ''
        print_value('-' * total_size)

    # Output Table
    top_headers = ['Ids(model_1, model_2)', 'Layers with ' + component_diffed + ' Differences']
    table_info_msg = "Note: Displays " + component_diffed + " differences between layers"
    col_sizes = [1 + len(header) for header in top_headers]
    col_sizes[0] = max(col_sizes[0], 6)
    if len(m1_dict): col_sizes[1] = max(max(col_sizes[1], max(list(map(len, m1_dict)))), 40)
    if len(m2_dict): col_sizes[1] = max(max(col_sizes[1], max(list(map(len, m2_dict)))), 40)

    # if our values are arrays we do not want to print the differences for each model, just the ids and names of
    # the layers with differences.

    # Required for dlcv4 to check for lists of ndarrays
    def is_arr_or_arrlist(val):
        return type(val) is np.ndarray or (type(val) is list and len(val) > 0 and type(val[0]) is np.ndarray)

    is_arr_val = any(is_arr_or_arrlist(val) for val, _ in m1_dict.values()) or \
                 any(is_arr_or_arrlist(val) for val, _ in m2_dict.values())

    if not is_arr_val:
        top_headers.append(input_dlc_one)
        top_headers.append(input_dlc_two)
        col_sizes.append(1 + len(input_dlc_one))
        col_sizes.append(1 + len(input_dlc_two))
        # find column sizes based on longest columns
        params1_max = 0
        params2_max = 0
        for layer_names, params1 in m1_dict.items():
            if params1[0] is not None and len(params1[0]) > 0:
                params1_max = max(params1_max, max(list(map(len, params1[0]))))
        for layer_names, params2 in m2_dict.items():
            if params2[0] is not None and len(params2[0]) > 0:
                params2_max = max(params2_max, max(list(map(len, params2[0]))))

        col_sizes[2] = max(max(col_sizes[2], 1 + params1_max), 40)
        col_sizes[3] = max(max(col_sizes[3], 1 + params2_max), 40)

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)

    print_value('', csv_content)
    print_value(table_info_msg, csv_content)
    print_value('-' * total_size)
    print_row(top_headers, col_sizes, csv_content)
    print_value('-' * total_size)

    model_with_extra_layers = ""
    num_extra_layers = 0
    if diff_strategy == DiffStrategies.BY_LAYER_IDS:
        m1_list = list(m1_dict.items())
        m2_list = list(m2_dict.items())
        # layers are ordered by ID so we can traverse as list
        diff_layers = shorter_len = len(m1_list) if len(m1_list) < len(m2_list) else len(m2_list)
        for i in range(shorter_len):
            ids = "(" + str(m1_list[i][1][1]) + ", " + str(m2_list[i][1][1]) + ")"
            m1_name, m2_name = m1_list[i][0], m2_list[i][0]
            m1_values, m2_values = m1_list[i][1][0], m2_list[i][1][0]
            _print_diffs(ids, m1_name, m2_name, m1_values, m2_values, is_arr_val, col_sizes, csv_content)

        # Print remaining layers unique to the larger model
        if len(m1_list) != shorter_len:
            model_with_extra_layers = input_dlc_one
            num_extra_layers = len(m1_list) - shorter_len
            for j in range(shorter_len, len(m1_list)):
                name, m1_values, id_ = m1_list[j][0], m1_list[j][1][0], str(m1_list[j][1][1])
                _print_diffs(id_, name, name, m1_values, [], is_arr_val, col_sizes, csv_content)
        elif len(m2_list) != shorter_len:
            model_with_extra_layers = input_dlc_two
            num_extra_layers = len(m2_list) - shorter_len
            for j in range(shorter_len, len(m2_list)):
                name, m2_values, id_ = m2_list[j][0], m2_list[j][1][0], str(m2_list[j][1][1])
                _print_diffs(id_, name, name, [], m2_values, is_arr_val, col_sizes, csv_content)
    else:
        # all diff values are among same layer names
        side_headers = list(m1_dict.keys())
        diff_layers = len(side_headers)
        for name in side_headers:
            m1_values = m1_dict.get(name)[0]
            m2_values = m2_dict.get(name)[0]
            ids = "(" + str(m1_dict.get(name)[1]) + ", " + str(m2_dict.get(name)[1]) + ")"
            _print_diffs(ids, name, name, m1_values, m2_values, is_arr_val, col_sizes, csv_content)

    # print extra details for analysis of model
    if num_shared_layers:
        if diff_layers == 0: diff_layers = 'none'
        if diff_layers == num_shared_layers: diff_layers = 'all'
        print_value("Info: Out of " + str(num_shared_layers) + " shared layers, " + str(diff_layers) +
                    " exhibit " + str(component_diffed) + " differences.", csv_content)
    if num_extra_layers:
        print_value("Info: Dlc model, " + str(model_with_extra_layers) + ", has " + str(num_extra_layers) +
                    " extra layers with " + str(component_diffed) + " values.", csv_content)


def compare_layers_by_name(m1, m2):
    """
    Compares layer names between two models.
    Stores layers with unique names/types for each model.

    @param m1: First DLC model
    @param m2: Second DLC model

    @return: A list containing a boolean variable, a list, and a string.

            Boolean:    True:   No unique layers
                        False:  Some unique layers

            List:       Contains two dictionaries
                        - Model 1 unique layers
                        - Model 2 unique layers

            String:     Info message for user
                        (will not appear when there is an output table)
    """

    # Dictionary of layer name and type info
    m1_types_info = m1.types_info()
    m2_types_info = m2.types_info()

    # Dictionary of layer name and id info
    m1_id_layer = m1.ids_layer()
    m2_id_layer = m2.ids_layer()

    # List of unique layers
    m1_unique_layers = list(set(m1_types_info.keys()) - set(m2_types_info.keys()))
    m2_unique_layers = list(set(m2_types_info.keys()) - set(m1_types_info.keys()))

    # Dictionary of unique layers, sorted by type, in one-to-many relationship
    m1_unique_sorted = {}
    m2_unique_sorted = {}

    for name, type in sorted(m1_types_info.items()):
        if name in m1_unique_layers:
            m1_unique_sorted.setdefault(type, []).append([name, m1_id_layer.get(name)])

    for name, type in sorted(m2_types_info.items()):
        if name in m2_unique_layers:
            m2_unique_sorted.setdefault(type, []).append([name, m2_id_layer.get(name)])

    # Check if Model 1 and 2 have all the same layers
    if len(m1_unique_layers) == 0 and len(m2_unique_layers) == 0:
        return [True, [m1_unique_sorted, m2_unique_sorted], '\nInfo: Model 1 and Model 2 share all of the '
                                                            'same layer names & types\n']
    else:
        return [False, [m1_unique_sorted, m2_unique_sorted], 'No output message, since there is an output table']


def compare_models_aix_records(m1_aix_records, m2_aix_records):
    """
    Compares and marks differences in aix meta information parameters for each record of two models.
    If record doesn't exist in one of the two models, marks the whole meta info in the record as new.

    @param m1_aix_records: First DLC model's record
    @param m2_aix_records: Second DLC model's record

    @return: A list containing a boolean variable, a list, and a string.

            Boolean:    True:   No Aix record and meta info difference between models
                        False:  Either a new Aix record in one of the models or shared records had meta info
                                differences

            List:       list of:
                        - Union of record names
                        - Dictionary of either unique aix records in model1 or aix records with unique meta info
                          in model1
                        - Dictionary of either unique aix records in model1 or aix records with unique meta info
                          in model2
                            * note: dictionary keys for these unique parameters are marked with asterisks

            String:     Info message for user
                        (will not appear when there is an output table)
    """

    m1_m2_aix_records = list(set().union(m1_aix_records, m2_aix_records))

    if m1_aix_records == m2_aix_records:
        # check if both models have aix enabled so that the correct
        # message is properly communicated
        if not m1_aix_records and not m2_aix_records:
            return [True, [m1_m2_aix_records, m1_aix_records, m2_aix_records],
                    '\nInfo: No HTA records found in both models\n']
        return [True, [m1_m2_aix_records, m1_aix_records, m2_aix_records],
                '\nInfo: HTA records are the same\n']

    m1_filtered_aix_record_dict = {}
    m2_filtered_aix_record_dict = {}
    m1_m2_aix_records = list(set().union(m1_aix_records, m2_aix_records))

    for record_key in m1_m2_aix_records:
        m1_filtered_aix_record_values = {}
        m2_filtered_aix_record_values = {}
        # mark all aix record values found in model11 but not model2
        if record_key in m1_aix_records and record_key not in m2_aix_records:
            m1_aix_meta_info = m1_aix_records[record_key]
            for info_key in m1_aix_meta_info:
                new_key = "*" + info_key  # add asterisks to key to indicate new param
                m1_filtered_aix_record_values[new_key] = m1_aix_meta_info[info_key]

        # mark all aix record values found in model2 but not model1
        elif record_key in m2_aix_records and record_key not in m1_aix_records:
            m2_aix_meta_info = m2_aix_records[record_key]
            for info_key in m2_aix_meta_info:
                new_key = "*" + info_key
                m2_filtered_aix_record_values[new_key] = m2_aix_meta_info[info_key]
        else:
            # this means record is in both, check difference in each parameter
            m1_aix_meta_info = m1_aix_records[record_key]
            m2_aix_meta_info = m2_aix_records[record_key]

            # check if any differences in each meta-info for records.
            if m1_aix_meta_info != m2_aix_meta_info:
                # mark meta keys unique to m1
                for m1_meta_info_key, m1_meta_info_value in m1_aix_meta_info.items():
                    if m1_meta_info_key not in m2_aix_meta_info or \
                                    m1_meta_info_value != m2_aix_meta_info[m1_meta_info_key]:
                        new_key = "*" + m1_meta_info_key
                        m1_aix_meta_info[new_key] = m1_aix_meta_info.pop(m1_meta_info_key)
                m1_filtered_aix_record_values.update(m1_aix_meta_info)

                # mark meta keys unique to m2
                for m2_meta_info_key, m2_meta_info_value in m2_aix_meta_info.items():
                    if m2_meta_info_key not in m1_aix_meta_info or \
                                    m2_meta_info_value != m1_aix_meta_info[m2_meta_info_key]:
                        new_key = "*" + m2_meta_info_key
                        m2_aix_meta_info[new_key] = m2_aix_meta_info.pop(m2_meta_info_key)
                m2_filtered_aix_record_values.update(m2_aix_meta_info)

        # add to corresponding dictionaries
        if m1_filtered_aix_record_values:
            m1_filtered_aix_record_dict[record_key] = m1_filtered_aix_record_values
        if m2_filtered_aix_record_values:
            m2_filtered_aix_record_dict[record_key] = m2_filtered_aix_record_values

    # get the aix record keys that diff
    m1_m2_aix_diff_records = list(set().union(m1_filtered_aix_record_dict, m2_filtered_aix_record_dict))
    return [False, [m1_m2_aix_diff_records, m1_filtered_aix_record_dict, m2_filtered_aix_record_dict],
            'No output message, since there is an output table']


def table_title(title_message, csv_content):
    """
    Header for output table
    """

    spaces = ' ' * 10
    col_sizes = [1 + len(header) for header in spaces]
    col_sizes[0] = max(col_sizes[0], 1 + 40)

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
    print_value('', csv_content)
    print_value('-' * total_size)
    print_value(title_message, csv_content)
    print_value('-' * total_size)


def print_general_table(m1_copyright_str, m2_copyright_str, m1_input_dims, m2_input_dims, m1_unique_sorted, m2_unique_sorted,
                        is_aix_enabled, m1_aix_record_present, m2_aix_record_present, m1_total_macs, m2_total_macs,
                        input_dlc_one, input_dlc_two, csv_content):

    """
    Top level table that compares input dimensions,
    number of unique layers between models, and total macs.

    @param m1_copyright_str: copyright applied when creating model1's dlc(if any)
    @param m2_copyright_str: copyright applied when creating model2's dlc(if any)
    @param m1_input_dims: Input dimensions of first dlc model
    @param m2_input_dims: Input dimensions of second dlc model
    @param m1_unique_sorted: Dictionary of unique layer names for first dlc model
    @param m2_unique_sorted: Dictionary of unique layer names for second dlc model
    @param is_aix_enabled: boolean value to indicate whether AIX is enabled in the build
    @param m1_aix_record_present: boolean string value to indicate whether dlc has Aix records
    @param m2_aix_record_present: boolean string value to indicate whether dlc has Aix records
    @param m1_total_macs: Total macs for first dlc model
    @param m2_total_macs: Total macs for second dlc model
    @param input_dlc_one: First DLC file name
    @param input_dlc_two: Second DLC file name
    """

    # Number of unique layers for each model
    m1_unique_len = 0
    m2_unique_len = 0
    for type, name_list in m1_unique_sorted.items():
        m1_unique_len += len(name_list)
    for type, name_list in m2_unique_sorted.items():
        m2_unique_len += len(name_list)

    m1_copyright = "Present"
    m2_copyright = "Present"
    if m1_copyright_str == 'N/A':
        m1_copyright = "Absent"
    if m2_copyright_str == 'N/A':
        m2_copyright = "Absent"

    # Output Table: General Differences
    sideheader1 = 'Model Copyrights'
    sideheader2 = 'Input Dimensions'
    sideheader3 = 'Unique layers (different layer names)'
    sideheader4 = 'HTA Enabled'
    sideheader5 = 'Total MACs'
    spaces = ' ' * max(len(sideheader1), len(sideheader2), len(sideheader3), len(sideheader4), len(sideheader5))

    topheaders = [spaces, input_dlc_one, input_dlc_two]
    col_sizes = [1 + len(header) for header in topheaders]

    col_sizes[0] = max(col_sizes[0], 1 + 30)
    col_sizes[1] = max(col_sizes[1], 1 + 30)
    col_sizes[2] = max(col_sizes[2], 1 + 30)

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)

    print_value('\nNote: Compares input dimensions, number of unique layers, and total macs', csv_content)
    print_value('-' * total_size)
    print_row(topheaders, col_sizes, csv_content)
    print_value('-' * total_size)
    print_row([sideheader1, m1_copyright, m2_copyright], col_sizes, csv_content)
    print_row([sideheader2, m1_input_dims, m2_input_dims], col_sizes, csv_content)
    print_row([sideheader3, m1_unique_len, m2_unique_len], col_sizes, csv_content)
    # AIX info is optional
    if is_aix_enabled:
        print_row([sideheader4, m1_aix_record_present, m2_aix_record_present], col_sizes, csv_content)
    print_row([sideheader5, get_si_notation(m1_total_macs, m1_total_macs),
               get_si_notation(m2_total_macs, m2_total_macs)], col_sizes, csv_content)
    print_value('-' * total_size)
    print_value('\n', csv_content)


def print_copyrights_table(m1_copyright_str, m2_copyright_str, input_dlc_one, input_dlc_two, csv_content):
    """
    Copyright differences table that display copyright statements for each dlc

    @param m1_copyright_str: copyright applied when creating model1's dlc(if any)
    @param m2_copyright_str: copyright applied when creating model2's dlc(if any)
    @param input_dlc_one: First DLC file name
    @param input_dlc_two: Second DLC file name
    @param csv_content: output file to optionally save diff output to
    """
    # Output Table: Copyright Differences Table
    top_headers = [input_dlc_one, input_dlc_two]
    col_sizes = [1 + len(header) for header in top_headers]

    # create list from copyright string by splitting it at every newline
    m1_copyright = m1_copyright_str.split('\n')
    m2_copyright = m2_copyright_str.split('\n')

    # find column sizes based on longest row/column pair
    # initial max column width
    m1_col_max = 15
    m2_col_max = 15

    for line in m1_copyright:
        m1_col_max = max(m1_col_max, len(line))
    for line in m2_copyright:
        m2_col_max = max(m2_col_max, len(line))

    col_sizes[0] = max(col_sizes[0], m1_col_max) + 3  # +3 for padding
    col_sizes[1] = max(col_sizes[1], m2_col_max) + 3  # +3 for padding

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
    print_value('-'*total_size, csv_content)
    print_row(top_headers, col_sizes, csv_content)
    print_value('-'*total_size, csv_content)

    # get the dlc with longer text
    max_num_lines = max(len(m1_copyright), len(m2_copyright))

    # print the copyright file contents for each dlc
    # We loop through the max_num_lines and update the column for each dlc if
    # it has lines left to print. if no more line, prints empty column
    for i in range(0, max_num_lines):
        col_list = ["", ""]  # initialize row as empty
        if i < len(m1_copyright):
            col_list[0] = m1_copyright[i].strip()
        if i < len(m2_copyright):
            col_list[1] = m2_copyright[i].strip()
        print_row(col_list, col_sizes, csv_content)

    print_value('-'*total_size, csv_content)


def print_layers_table(m1_unique_sorted, m2_unique_sorted, input_dlc_one, input_dlc_two, csv_content):
    """
    Layer Differences table that displays the unique layers for each model.
    Outputs layers that are present in one model, and absent from the other.

    @param m1_unique_sorted: Dictionary of unique layer names for first dlc model
    @param m2_unique_sorted: Dictionary of unique layer names for second dlc model
    @param input_dlc_one: First DLC file name
    @param input_dlc_two: Second DLC file name
    """

    # Output Table: Layer Differences Table
    sideheaders1 = list(m1_unique_sorted.keys())
    sideheaders2 = list(m2_unique_sorted.keys())

    topheaders1 = ['Id', 'Layers unique to Network#1 not present in Network#2 (organized by type)', input_dlc_one, input_dlc_two]
    topheaders2 = ['Id', 'Layers unique to Network#2 not present in Network#1 (organized by type)', input_dlc_one, input_dlc_two]
    col_sizes = [1 + len(header) for header in topheaders1]

    # get the max length for the layer type names to determine column size
    if not m1_unique_sorted:
        max_layer_type_name = max(list(map(len, m2_unique_sorted)))
    elif not m2_unique_sorted:
        max_layer_type_name = max(list(map(len, m1_unique_sorted)))
    else:
        max_layer_type_name = max(max(list(map(len, m1_unique_sorted))), max(list(map(len, m2_unique_sorted))))

    # get the max length for the layer names to determine column size
    max_layer_name = 0
    if not m1_unique_sorted:
        for m in m2_unique_sorted.values():
            max_layer_name = max(max_layer_name, max(len(x[0]) for x in m))
    elif not m2_unique_sorted:
        for m in m1_unique_sorted.values():
            max_layer_name = max(max_layer_name, max(len(x[0]) for x in m))
    else:
        for m in m1_unique_sorted.values():
            max_layer_name = max(max_layer_name, max(len(x[0]) for x in m))
        for m in m2_unique_sorted.values():
            max_layer_name = max(max_layer_name, max(len(x[0]) for x in m))

    col_sizes[0] = max(col_sizes[0], 6)
    col_sizes[1] = max(col_sizes[1], 1 + 6 + max_layer_type_name, 2 + max_layer_name, 1 + 40)
    col_sizes[2] = max(col_sizes[2], 1 + 40)
    col_sizes[3] = max(col_sizes[3], 1 + 40)

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
    print_value('\nNote: Displays layer differences between models, organized by layer type', csv_content)
    print_value('-' * total_size)
    print_row(topheaders1, col_sizes, csv_content)
    print_value('-' * total_size)

    for type in sideheaders1:
        print_row(['','type: ' + type, '', ''], col_sizes, csv_content)
        for name_id_list in m1_unique_sorted.get(type):
            print_row([name_id_list[1], ('(%s)' % (name_id_list[0])), 'Present', 'Absent'], col_sizes, csv_content)
        print_value('-' * total_size)

    print_value('-' * total_size)
    print_row(topheaders2, col_sizes, csv_content)
    print_value('-' * total_size)

    for type in sideheaders2:
        print_row(['','type: ' + type, '', ''], col_sizes, csv_content)
        for name_id_list in m2_unique_sorted.get(type):
            print_row([name_id_list[1], ('(%s)' % (name_id_list[0])), 'Absent', 'Present'], col_sizes, csv_content)
        print_value('-' * total_size)
    print_value('', csv_content)


def supports_color():
    """
    Returns True if the running system's terminal supports color, and False
    otherwise.
    """
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or
                                                  'ANSICON' in os.environ)
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    if not supported_platform or not is_a_tty:
        return False
    return True


def prCyan(skk):
    """
    Returns string in cyan color.
    """
    colored_skk ="\033[96m{}\033[00m".format(skk)
    return colored_skk


def prRed(skk):
    """
    Returns string in red color.
    """
    colored_skk = "\033[91m{}\033[00m" .format(skk)
    return colored_skk


def print_row_value_table(values, col_sizes, offset, csv_content):
    """
    Print rows by highlight varying values in a shared layer with color.
    """
    print('|', end=' ')
    for value, size in zip(values, col_sizes):
        if value and value != values[0] and value != values[1] and supports_color:
            value = prCyan(value)
            size += offset
        print('{0:<{1}}|'.format(value, size), end=' ')
    print()
    if snpe_dlc_utils.csv_file_flag:
        csv_content.append(values)


def print_aix_param_row(col_list, col_sizes, csv_content):
    """
    Prints rows separated by '|' character. If the asterisks character exist
    in column it will try to print the column in different color, if supported by terminal

    @param col_list: row of columns to print
    @param col_sizes: size for each column
    @param csv_content: output file to optionally save diff output to
    """

    terminal_color_support = supports_color()
    local_col_size = []  # separate local variable so that we dont modify outer scope col_sizes
    if terminal_color_support:
        offset = len(prCyan("hello"))-len("hello") # need to re-adjust since adding color introduces offset
        for i in range(0, len(col_list)):
            if "[96m" in col_list[i]:  # already has cyan color
                local_col_size.append(col_sizes[i] + offset)
            elif "*" in col_list[i]:
                # change color of key to reflect param difference
                col_list[i] = prCyan(col_list[i])
                local_col_size.append(col_sizes[i] + offset)
            else:
                local_col_size.append(col_sizes[i])
    else:
        local_col_size = col_sizes

    print_row(col_list, local_col_size, csv_content)


def add_aix_diff_columns(col_list, col_val, current_col_num, current_col_row_num, reference_col_row_num, add_color):
    """
        Adds columns for the aix diff table. This helps to align the 2 columns for models.
        Note: it only supports rows with 3 columns and aligns columns 2 and 3
        @param col_list: the general list that will be used to print the table rows
        @param col_val: the current column content
        @param current_col_num: the current column number.
        @param current_col_row_num: the number of rows that have been so far added for current column
        @param reference_col_row_num: the neighbouring column to align too. This helps to decide whether a row has
                                      already been created for that column
        @param add_color: boolean to determine if value is unique in model and must be colored to visually show
                         difference. True if must be colored, False otherwise
    """
    if add_color:
        col_val = prCyan(col_val)
    # only add a new row if current column doesnt have empty rows
    if current_col_row_num >= reference_col_row_num:
        if current_col_num == 1:
            col_list.append(["", col_val, ""])
        elif current_col_num == 2:
            col_list.append(["", "", col_val])
    else:
        col_list[current_col_row_num][current_col_num] = col_val
    current_col_row_num += 1

    return current_col_row_num


def print_aix_table(aix_record_keys, m1_aix_record_dict, m2_aix_record_dict, input_dlc_one, input_dlc_two,
                    aix_error_messages, csv_content):
    """
    HTA Differences table that displays HTA record and meta information differences
    between the two models.

    @param aix_record_keys: List of the union of record names between the two models
    @param m1_aix_record_dict: Dictionary of unique records in model1
    @param m2_aix_record_dict: Dictionary of unique records in model2
    @param input_dlc_one: First DLC file name
    @param input_dlc_two: Second DLC file name
    @param aix_error_messages: dictionary with key as dlc file name and value is the error message when querying the dlc
    @param csv_content: output file to optionally save diff output to
    """

    # Output Table: Aix Records Differences Table
    top_headers = ['HTA Records with differences', input_dlc_one, input_dlc_two]
    col_sizes = [1 + len(header) for header in top_headers]

    # find column sizes based on longest row/column pair
    # initial min column width
    record_key_col_max = 0
    m1_col_max = 0
    m2_col_max = 0
    for record_name in aix_record_keys:
        record_key_col_max = max(record_key_col_max, len(record_name))
    for record_name, meta_info in m1_aix_record_dict.items():
        m1_col_max = max(m1_col_max, len(max(map(str, meta_info.values()), key=len)))
    for record_name, meta_info in m2_aix_record_dict.items():
        m2_col_max = max(m2_col_max, len(max(map(str, meta_info.items()), key=len)))

    # 70 for upper limit to column characters per line
    max_col_size = 70
    m1_col_max = min(max_col_size, m1_col_max)
    m2_col_max = min(max_col_size, m2_col_max)

    col_sizes[0] = max(col_sizes[0], record_key_col_max)
    col_sizes[1] = max(col_sizes[1], m1_col_max)
    col_sizes[2] = max(col_sizes[2], m2_col_max)

    total_size = 2 + 2 * len(col_sizes) - 1 + sum(col_sizes)
    print_value('-'*total_size)
    print_row(top_headers, col_sizes, csv_content)
    print_value('-'*total_size)

    # column initial values. This helps to determine if the record is not present in the model
    # or if there was an error loading the model's aix records
    m1_col_initial_val = "Not Present"
    m2_col_initial_val = "Not Present"
    # assign columns as Query Errors if error due to reasons other than aix compatibility
    if input_dlc_one in aix_error_messages.keys() and "aix_compatibility" not in aix_error_messages[input_dlc_one].values():
        m1_col_initial_val = "HTA Query Error"
    if input_dlc_two in aix_error_messages.keys() and "aix_compatibility" not in aix_error_messages[input_dlc_two].values():
        m2_col_initial_val = "HTA Query Error"

    for record_key in aix_record_keys:
        m1_aix_meta_dict = {}
        m2_aix_meta_dict = {}

        # first print aix record name and whether if present in models
        col_list = [record_key, m1_col_initial_val, m2_col_initial_val]
        if record_key in m1_aix_record_dict:
            col_list[1] = "Params:"
            m1_aix_meta_dict = m1_aix_record_dict[record_key]
        if record_key in m2_aix_record_dict:
            col_list[2] = "Params:"
            m2_aix_meta_dict = m2_aix_record_dict[record_key]
        print_row(col_list, col_sizes, csv_content)

        # next print the parameter details for aix records for each model
        m1_m2_aix_meta_keys = list(set().union(m1_aix_meta_dict.keys(), m2_aix_meta_dict.keys()))

        col_list = []
        # to keep track of rows for each column
        m1_col_row = 0
        m2_col_row = 0
        add_color = False
        for aix_key in m1_m2_aix_meta_keys:
            if aix_key in m1_aix_meta_dict:
                if "*" in aix_key:
                    add_color = True  # if the key/value pair is marked as unique, color key and all the values as well
                col_val = m1_aix_meta_dict[aix_key]
                # if sub-dict present print on separate row for better visual
                # note: only supports 1 level deep
                if isinstance(col_val, dict):
                    m1_col_row = add_aix_diff_columns(col_list, "  " + aix_key + ":", 1, m1_col_row, m2_col_row, add_color)

                    for meta_key, meta_values in m1_aix_meta_dict[aix_key].items():
                        # again if list present in value print on separate row for better visual
                        # note: added to support printing buffer input/output on separate lines
                        if not isinstance(meta_values, list):
                            m1_col_row = add_aix_diff_columns(col_list, "    " + meta_key + ":" + str(meta_values), 1, m1_col_row, m2_col_row, add_color)
                        else:
                            m1_col_row = add_aix_diff_columns(col_list, "    " + meta_key + ":", 1, m1_col_row, m2_col_row, add_color)
                            for v in meta_values:
                                m1_col_row = add_aix_diff_columns(col_list, "      " + v, 1, m1_col_row, m2_col_row, add_color)
                else:
                    m1_col_row = add_aix_diff_columns(col_list, "  " + aix_key + ":" + str(col_val), 1, m1_col_row, m2_col_row, add_color)
                add_color = False  # reset coloring for next key
            if aix_key in m2_aix_meta_dict:
                if "*" in aix_key:
                    add_color = True
                col_val = m2_aix_meta_dict[aix_key]
                if isinstance(col_val, dict):
                    m2_col_row = add_aix_diff_columns(col_list, "  " + aix_key + ":", 2, m2_col_row, m1_col_row, add_color)

                    for meta_key, meta_values in m2_aix_meta_dict[aix_key].items():
                        if not isinstance(meta_values, list):
                            m2_col_row = add_aix_diff_columns(col_list, "    " + meta_key + ":" + str(meta_values),2, m2_col_row, m1_col_row, add_color)
                        else:
                            m2_col_row = add_aix_diff_columns(col_list, "    " + meta_key + ":", 2, m2_col_row, m1_col_row, add_color)
                            for v in meta_values:
                                m2_col_row = add_aix_diff_columns(col_list, "      " + v, 2, m2_col_row, m1_col_row, add_color)
                else:
                    m2_col_row = add_aix_diff_columns(col_list, "  " + aix_key + ":" + str(col_val), 2, m2_col_row, m1_col_row, add_color)
                add_color = False
        for cols in col_list:
            print_aix_param_row(cols, col_sizes, csv_content)
        print_value('-' * total_size)

    print_value('', csv_content)


class AixCompatibilityException(Exception):
    pass


def check_incompatible_aix_records(aix_records):
    warning_msgs = ""
    for aix_record_name, aix_meta_info in aix_records.items():
        if not aix_meta_info['compatibility']:
            warning_msgs += "- Record " + aix_record_name + " is incompatible with the latest version of SNPE\n"
    if len(warning_msgs):
        raise AixCompatibilityException(warning_msgs)


def _get_and_match_tensors(m1, m2, diff_strategy):
    """
    Performs tensor gathering, layer pairing, and tensor matching once.
    Returns a dictionary with all the data needed for tensor type comparisons.
    """
    m1_tensor_info = {}
    m2_tensor_info = {}

    first_graph_m1 = list(m1.rows.keys())[0]
    first_graph_m2 = list(m2.rows.keys())[0]
    # 1. Gather tensor info
    for row in m1.rows[first_graph_m1]:
        for t in list(row.op.inputs()) + list(row.op.outputs()):
            if t.name() not in m1_tensor_info:
                m1_tensor_info[t.name()] = {
                    'layer_name': row.op.name,
                    'layer_id'  : row.id,
                    'tensor_type': t.tensor_type()
                }

    for row in m2.rows[first_graph_m2]:
        for t in list(row.op.inputs()) + list(row.op.outputs()):
            if t.name() not in m2_tensor_info:
                m2_tensor_info[t.name()] = {
                    'layer_name': row.op.name,
                    'layer_id'  : row.id,
                    'tensor_type': t.tensor_type()
                }

    # 2. Pair layers
    shared_layer_pairs = []
    if diff_strategy == DiffStrategies.BY_LAYER_IDS:
        m1_sorted = m1.rows[first_graph_m1]
        m2_sorted = m2.rows[first_graph_m2]
        num_to_match = min(len(m1_sorted), len(m2_sorted))
        for i in range(num_to_match):
            shared_layer_pairs.append((m1_sorted[i], m2_sorted[i]))
    else:
        m1_by_name = {row.op.name: row for row in m1.rows[first_graph_m1]}
        m2_by_name = {row.op.name: row for row in m2.rows[first_graph_m2]}
        common_names = set(m1_by_name.keys()) & set(m2_by_name.keys())
        for name in sorted(common_names):
            shared_layer_pairs.append((m1_by_name[name], m2_by_name[name]))

    # 3. Match tensors
    shared_tensors = {}
    m1_unmatched = set(m1_tensor_info.keys())
    m2_unmatched = set(m2_tensor_info.keys())

    for layer1, layer2 in shared_layer_pairs:
        out1 = list(layer1.op.outputs())
        out2 = list(layer2.op.outputs())
        num_to_match = min(len(out1), len(out2))
        for i in range(num_to_match):
            t1_name = out1[i].name()
            t2_name = out2[i].name()
            if t1_name in m1_unmatched and t2_name in m2_unmatched:
                shared_tensors[t1_name] = t2_name
                m1_unmatched.remove(t1_name)
                m2_unmatched.remove(t2_name)

    common_unmatched_names = m1_unmatched & m2_unmatched
    for name in sorted(common_unmatched_names):
        shared_tensors[name] = name
        m1_unmatched.remove(name)
        m2_unmatched.remove(name)

    return {
        "m1_tensor_info": m1_tensor_info,
        "m2_tensor_info": m2_tensor_info,
        "shared_tensors": shared_tensors,
        "m1_unmatched": m1_unmatched,
        "m2_unmatched": m2_unmatched,
        "shared_layer_count": len(shared_layer_pairs)
    }


# --------------------------------------------------------------
#  print_output_tensor_type_diffs
# --------------------------------------------------------------
def print_output_tensor_type_diffs(tensor_data, dlc_one_base, dlc_two_base,
                                  csv_content):
    """
    Print two tables:

    1.  Matched tensors whose **output tensor type differs** between
        the two models.
    2.  Tensors that exist only in one model (unique).

    Parameters
    ----------
    tensor_data       : Dict with pre-computed tensor comparison data.
    dlc_one_base,
    dlc_two_base      : Base file names - used for column titles.
    csv_content       : List that receives CSV rows (used by the helper
                        ``print_value``/``print_row``).
    """
    m1_tensor_info = tensor_data["m1_tensor_info"]
    m2_tensor_info = tensor_data["m2_tensor_info"]
    shared_tensors = tensor_data["shared_tensors"]
    m1_unmatched = tensor_data["m1_unmatched"]
    m2_unmatched = tensor_data["m2_unmatched"]

    # ------------------------------------------------------------------
    # Build the rows that will be printed.
    # ------------------------------------------------------------------
    diff_rows      = []      # shared tensors with different type
    unmatched_rows = []      # tensors that belong to only one model

    # ---- shared-but-different ----
    for m1_name, m2_name in sorted(shared_tensors.items()):
        t1_type = m1_tensor_info[m1_name]['tensor_type']
        t2_type = m2_tensor_info[m2_name]['tensor_type']
        if t1_type != t2_type:
            m1_layer_name = m1_tensor_info[m1_name]['layer_name']
            m2_layer_name = m2_tensor_info[m2_name]['layer_name']
            ids = (f"({m1_tensor_info[m1_name]['layer_id']},"
                   f"{m2_tensor_info[m2_name]['layer_id']})")
            t1_str = f"{m1_name}: {t1_type}"
            t2_str = f"{m2_name}: {t2_type}"

            if m1_layer_name == m2_layer_name:
                row = [
                    ids,
                    m1_layer_name,
                    t1_str,
                    t2_str
                ]
                diff_rows.append(row)
            else:
                diff_rows.append([
                    ids,
                    m1_layer_name,
                    t1_str,
                    ""
                ])
                diff_rows.append([
                    "",
                    m2_layer_name,
                    "",
                    t2_str
                ])

    # ---- unmatched tensors ----
    for name in sorted(m1_unmatched):
        info = m1_tensor_info[name]
        row = [
            f"({info['layer_id']}, N/A)",
            info['layer_name'],
            f"{name}: {info['tensor_type']}",
            "Missing"
        ]
        unmatched_rows.append(row)

    for name in sorted(m2_unmatched):
        info = m2_tensor_info[name]
        row = [
            f"(N/A, {info['layer_id']})",
            info['layer_name'],
            "Missing",
            f"{name}: {info['tensor_type']}"
        ]
        unmatched_rows.append(row)

    # ------------------------------------------------------------------
    # Print the tables
    # ------------------------------------------------------------------
    if not diff_rows and not unmatched_rows:
        print_value("No differences found in output tensor types.", csv_content)
        print_value("", csv_content)
        return

    # ---- Table 1 - mismatched types ----
    if diff_rows:
        table_title('Comparable tensors with Different Output Tensor Types', csv_content)

        # column titles
        headers = [
            f'Ids({dlc_one_base},{dlc_two_base})',
            'Layer Name',
            f'Tensor ({dlc_one_base})',
            f'Tensor ({dlc_two_base})'
        ]
        # initial column widths = header length + 1 (for a little padding)
        col_sizes = [len(h) + 1 for h in headers]

        # enlarge column widths to fit the data
        for row in diff_rows:
            for i, cell in enumerate(row):
                col_sizes[i] = max(col_sizes[i], len(str(cell)) + 1)

        total_width = sum(col_sizes) + len(col_sizes) * 3 - 1
        print_row(headers, col_sizes, csv_content)
        print_value('-' * total_width, csv_content)
        for row in diff_rows:
            print_row(row, col_sizes, csv_content)
        print_value('-' * total_width, csv_content)
        print_value("", csv_content)

    # ---- Table 2 - unmatched tensors ----
    if unmatched_rows:
        table_title('Unique Tensors', csv_content)

        headers = [
            f'Ids({dlc_one_base},{dlc_two_base})',
            'Layer Name',
            f'Tensor ({dlc_one_base})',
            f'Tensor ({dlc_two_base})'
        ]
        col_sizes = [len(h) + 1 for h in headers]

        for row in unmatched_rows:
            for i, cell in enumerate(row):
                col_sizes[i] = max(col_sizes[i], len(str(cell)) + 1)

        total_width = sum(col_sizes) + len(col_sizes) * 3 - 1
        print_row(headers, col_sizes, csv_content)
        print_value('-' * total_width, csv_content)
        for row in unmatched_rows:
            print_row(row, col_sizes, csv_content)
        print_value('-' * total_width, csv_content)
        print_value("", csv_content)


# --------------------------------------------------------------
#  compare_output_tensor_types
# --------------------------------------------------------------
def compare_output_tensor_types(m1, m2, diff_strategy):
    """
    Returns a tuple that the main driver expects:

    (boolean, data_dict, message_string, shared_layer_count)
    """
    tensor_data = _get_and_match_tensors(m1, m2, diff_strategy)
    m1_tensor_info = tensor_data["m1_tensor_info"]
    m2_tensor_info = tensor_data["m2_tensor_info"]

    if tensor_data["m1_unmatched"] or tensor_data["m2_unmatched"]:
        return (False,
                tensor_data,
                "Output tensor types have differences (missing tensors)",
                tensor_data["shared_layer_count"])

    for t1, t2 in tensor_data["shared_tensors"].items():
        if m1_tensor_info[t1]['tensor_type'] != m2_tensor_info[t2]['tensor_type']:
            return (False,
                    tensor_data,
                    "Output tensor types have differences (type mismatch)",
                    tensor_data["shared_layer_count"])

    return (True,
            tensor_data,
            "Output tensor types are the same",
            tensor_data["shared_layer_count"])


def print_output_tensor_types_summary(
        tensor_data,
        dlc_one_base, dlc_two_base,
        csv_content):
    """
    Summarise tensor-type differences between two DLCs.

    Two tables are printed:

    1.  Summary table (4 numbers)
        • Shared tensors (same type)
        • Shared tensors (different type)
        • Non-shared tensors in model-1
        • Non-shared tensors in model-2

    2.  Cross-type matrix
        rows   = tensor types that appear in model-1
        columns= tensor types that appear in model-2
        each cell = number of Comparable tensors whose type in model-1 is the row-type
                    and whose type in model-2 is the column-type
        (diagonal = “same-type” count).

    Parameters
    ----------
    tensor_data            : Dict with pre-computed tensor comparison data.
    dlc_one_base, dlc_two_base : Base file names - used for column titles.
    csv_content            : list that receives CSV rows (used by the helper
                             ``print_value``/``print_row``).
    """
    m1_tensor_info = tensor_data["m1_tensor_info"]
    m2_tensor_info = tensor_data["m2_tensor_info"]
    shared_tensors = tensor_data["shared_tensors"]
    unmatched_m1 = tensor_data["m1_unmatched"]
    unmatched_m2 = tensor_data["m2_unmatched"]

    # ------------------------------------------------------------------
    # Compute the statistics required for the two tables.
    # ------------------------------------------------------------------
    # ---- Table 1 counters ------------------------------------------------
    same_type_cnt = 0
    diff_type_cnt = 0

    for t1_name, t2_name in shared_tensors.items():
        t1_type = m1_tensor_info[t1_name]['tensor_type']
        t2_type = m2_tensor_info[t2_name]['tensor_type']
        if t1_type == t2_type:
            same_type_cnt += 1
        else:
            diff_type_cnt += 1

    non_shared_m1_cnt = len(unmatched_m1)
    non_shared_m2_cnt = len(unmatched_m2)

    # ---- Table 2 matrix --------------------------------------------------
    # rows = tensor types that appear in model 1
    # cols = tensor types that appear in model 2
    all_types_m1 = sorted({info['tensor_type'] for info in m1_tensor_info.values()})
    all_types_m2 = sorted({info['tensor_type'] for info in m2_tensor_info.values()})

    # initialise matrix with zeros
    matrix = {rt: {ct: 0 for ct in all_types_m2} for rt in all_types_m1}

    for t1_name, t2_name in shared_tensors.items():
        rt = m1_tensor_info[t1_name]['tensor_type']
        ct = m2_tensor_info[t2_name]['tensor_type']
        matrix[rt][ct] += 1

    # ------------------------------------------------------------------
    # Print Table 1 - the compact summary.
    # ------------------------------------------------------------------
    table_title('Output Tensor Type Differences Summary', csv_content)

    headers1 = [
        "Comparable tensors (Same Type)",
        "Comparable tensors (Diff Type)",
        f"Unique tensors (in {dlc_one_base})",
        f"Unique tensors (in {dlc_two_base})"
    ]
    col_w1 = [len(h) for h in headers1]
    values1 = [same_type_cnt, diff_type_cnt, non_shared_m1_cnt, non_shared_m2_cnt]
    for i, v in enumerate(values1):
        col_w1[i] = max(col_w1[i], len(str(v)))

    header_line1 = " | ".join(f"{h:<{col_w1[i]}}" for i, h in enumerate(headers1))
    value_line1  = " | ".join(f"{str(values1[i]):<{col_w1[i]}}" for i in range(len(values1)))

    print_value(header_line1, csv_content)
    print_value("-" * len(header_line1), csv_content)
    print_value(value_line1, csv_content)
    print_value("", csv_content)

    # ------------------------------------------------------------------
    # Print Table 2 - cross-type matrix.
    # ------------------------------------------------------------------
    caption = (
        f"Cross-tabulation of comparable tensor types between {dlc_one_base} and {dlc_two_base}. "
        f"Diagonal values indicate same tensor types, while off-diagonal "
        f"values show changed tensor types."
    )

    print_value(caption, csv_content)
    print_value("", csv_content)
    col_widths = []

    first_col_width = max(len(f"{dlc_one_base}\{dlc_two_base}"),
                          max((len(rt) for rt in all_types_m1), default=0))
    col_widths.append(first_col_width)

    # width for each model-2 type column - max of the header and the numbers underneath
    for ct in all_types_m2:
        max_num_width = max(len(str(matrix[rt][ct])) for rt in all_types_m1) if all_types_m1 else 0
        col_widths.append(max(len(ct), max_num_width))

    # ---- Header row ----------------------------------------------------
    header_cells = [f"{dlc_one_base}\{dlc_two_base}"] + all_types_m2
    header_line2 = " | ".join(f"{header_cells[i]:<{col_widths[i]}}" for i in range(len(header_cells)))
    print_value(header_line2, csv_content)
    print_value("-" * len(header_line2), csv_content)

    # ---- Data rows ----------------------------------------------------
    for rt in all_types_m1:
        row_cells = [rt] + [str(matrix[rt][ct]) for ct in all_types_m2]
        line = " | ".join(f"{row_cells[i]:<{col_widths[i]}}" for i in range(len(row_cells)))
        print_value(line, csv_content)

    print_value("", csv_content)

def display_dlc_differences(m1, m2, csv_content, args, sdk):
    """
    Captures the "difference" results from model comparison functions above.

    Based on these results and command line arguments provided by the user,
    outputs informative messages or "difference" tables.

    Determines which help messages to provide the user
    (e.g. See --parameters or --enable_parameters for more details)

    @param m1: First DLC's model info
    @param m2: Second DLC's model info
    @param csv_content: output file to optionally save diff output to
    @param args: arguments from the user
    """
    dlc_compare_criteria = []  # List to hold all criteria for comparing the DLCs

    # Extract model information layer by layer
    for temp in m1.graph_names:
        global graphName1
        graphName1 = temp
    m1.extract_model_info(graphName1)

    for temp in m2.graph_names:
        global graphName2
        graphName2 = temp
    m2.extract_model_info(graphName2)

    copyrights = False
    layers = False
    parameters = False
    dimensions = False
    weights = False
    outputs = False
    output_tensor_types = False
    diff_by_id = False
    hta = False

    if sdk == "qairt":
        copyrights_str = "--compare_copyrights"
        layers_str = "--compare_layers"
        parameters_str = "--compare_parameters"
        dimensions_str = "--compare_dimensions"
        weights_str = "--compare_weights"
        outputs_str = "--compare_outputs"
        output_tensor_types_str = "--compare_output_tensor_types"
        diff_by_id_str = "--enable_diff_by_id"
        hta_str = "--compare_hta"

        if args.compare_copyrights:
            copyrights = True
        if args.compare_layers:
            layers = True
        if args.compare_parameters:
            parameters = True
        if args.compare_dimensions:
            dimensions = True
        if args.compare_weights:
            weights = True
        if args.compare_outputs:
            outputs = True
        if args.compare_output_tensor_types:
            output_tensor_types = True
        if args.enable_diff_by_id:
            diff_by_id = True
        if m1.is_aix_enabled() and args.compare_hta:
            hta = True

    elif sdk == "snpe":
        if args.copyrights:
            copyrights = True
        if args.layers:
            layers = True
        if args.parameters:
            parameters = True
        if args.dimensions:
            dimensions = True
        if args.weights:
            weights = True
        if args.outputs:
            outputs = True
        if args.diff_by_id:
            diff_by_id = True
        if m1.is_aix_enabled() and args.hta:
            hta = True
        if args.compare_output_tensor_types:
            output_tensor_types = True

    # Retrieve base names of DLC files
    dlc_one_base = os.path.splitext(os.path.basename(args.input_dlc_one))[0]
    dlc_two_base = os.path.splitext(os.path.basename(args.input_dlc_two))[0]

    # Input dimensions for Models 1 and 2
    m1_input_dims = m1.get_input_dims()
    m2_input_dims = m2.get_input_dims()

    # Total MACs for Models 1 and 2
    m1_total_macs = m1.get_total_macs(graphName1)
    m2_total_macs = m2.get_total_macs(graphName2)

    # get copyright infos
    m1_copyright_str = m1.get_model_copyright()
    m2_copyright_str = m2.get_model_copyright()
    copyrights_same = False
    if m1_copyright_str == m2_copyright_str:
        copyrights_same = True
    dlc_compare_criteria.append(copyrights_same)

    # Capture layer, parameter, dimension, weight, outputs... information #

    # set diff strategies for comparing values based on argument passed passed.
    diff_strategy = DiffStrategies.BY_COMMON_NAMES
    if diff_by_id:
        diff_strategy = DiffStrategies.BY_LAYER_IDS

    layer_comparison_result = compare_layers_by_name(m1, m2)
    parameter_comparison_result = get_layers_with_value_diffs(m1, m2, snpe_dlc_utils.ModelInfo.params_info,
                                                              "Parameters", diff_strategy)
    dimension_comparison_result = get_layers_with_value_diffs(m1, m2, snpe_dlc_utils.ModelInfo.dims_info, "Dimensions",
                                                              diff_strategy)
    output_names_comparison_results = get_layers_with_value_diffs(m1, m2, snpe_dlc_utils.ModelInfo.get_output_names,
                                                                  "Output", diff_strategy)
    weight_comparison_result = get_layers_with_value_diffs(m1, m2, snpe_dlc_utils.ModelInfo.weights_info,
                                                           "Weight", diff_strategy)
    output_tensor_types_comparison_result = compare_output_tensor_types(m1, m2, diff_strategy)
    # Truth value that represents how Models 1 and 2 are related (True if they are the same)
    layers_same = layer_comparison_result[0]
    dlc_compare_criteria.append(layers_same)

    parameters_same = parameter_comparison_result[0]
    dlc_compare_criteria.append(parameters_same)

    dimensions_same = dimension_comparison_result[0]
    dlc_compare_criteria.append(dimensions_same)

    output_names_same = output_names_comparison_results[0]
    dlc_compare_criteria.append(output_names_same)

    weights_same = weight_comparison_result[0]
    dlc_compare_criteria.append(weights_same)

    output_tensor_types_same = output_tensor_types_comparison_result[0]
    dlc_compare_criteria.append(output_tensor_types_same)

    # Dictionary of layers that exhibit differences between Models 1 and 2
    [m1_unique_sorted, m2_unique_sorted] = layer_comparison_result[1]
    [m1_diff_parameters, m2_diff_parameters] = parameter_comparison_result[1]
    [m1_diff_dimensions, m2_diff_dimensions] = dimension_comparison_result[1]
    [m1_diff_output_names, m2_diff_output_names] = output_names_comparison_results[1]
    [m1_diff_weights, m2_diff_weights] = weight_comparison_result[1]
    tensor_comparison_data = output_tensor_types_comparison_result[1]

    # Info message from comparison
    layers_message = layer_comparison_result[2]
    parameters_message = parameter_comparison_result[2]
    dimensions_message = dimension_comparison_result[2]
    output_names_message = output_names_comparison_results[2]
    weights_message = weight_comparison_result[2]
    output_tensor_types_message = output_tensor_types_comparison_result[2]

    # Total number of shared layers
    param_num_shared_layers = parameter_comparison_result[3]
    dim_num_shared_layers = dimension_comparison_result[3]
    output_names_num_shared_layers = output_names_comparison_results[3]
    weights_num_shared_layers = weight_comparison_result[3]
    output_tensor_types_num_shared_layers = output_tensor_types_comparison_result[3]

    # Capture Aix Record information
    m1_aix_records = {}
    m2_aix_records = {}
    # the actual model doesn't matter as both are compiled with the same DnnModel object
    is_aix_enabled = m1.is_aix_enabled()
    m1_aix_record_present = "False"
    m2_aix_record_present = "False"

    if is_aix_enabled:

        # Check if models have aix enabled
        if m1.is_aix_record_present():
            m1_aix_record_present = "True"
        if m2.is_aix_record_present():
            m2_aix_record_present = "True"

        # check if models have same aix records
        aix_error_messages = {}
        try:
            m1_aix_records = m1.get_aix_records()
            check_incompatible_aix_records(m1_aix_records)
        except AixCompatibilityException as e:
            aix_error_messages.update({dlc_one_base:
                                      {"msg": "***Error: Querying HTA Records for model: " + m1.model_filename + ":\n" + e.message,
                                       "err_type": "aix_compatibility"}})
        except Exception as e:
            aix_error_messages.update({dlc_one_base:
                                       {"msg": "***Error: Querying HTA Records for model: " + m1.model_filename + ":\n" + e.message,
                                        "err_type": "other"}})
        try:
            m2_aix_records = m2.get_aix_records()
            check_incompatible_aix_records(m2_aix_records)
        except AixCompatibilityException as e:
            aix_error_messages.update({dlc_two_base:
                                      {"msg": "***Error: Querying HTA Records for model: " + m2.model_filename + ":\n" + e.message,
                                       "err_type": "aix_compatibility"}})
        except Exception as e:
            aix_error_messages.update({dlc_two_base:
                                      {"msg": "***Error: Querying HTA Records for model: " + m2.model_filename + "\n" + e.message,
                                       "err_type": "other"}})

        aix_comparison_result = compare_models_aix_records(m1_aix_records, m2_aix_records)
        aix_records_same = aix_comparison_result[0]
        dlc_compare_criteria.append (aix_records_same)
        aix_record_keys, m1_aix_record_dict, m2_aix_record_dict = aix_comparison_result[1]
        aix_records_message = aix_comparison_result[2]

    # Check if weights, params and AIP record contents are identical. If so, DLCs are the same.
    if all(dlc_compare_criteria):
        print_value("Note: DLCs are the same", csv_content)
    else:
        table_title('General Table', csv_content)
        print_general_table(m1_copyright_str, m2_copyright_str, m1_input_dims, m2_input_dims, m1_unique_sorted, m2_unique_sorted,
                            is_aix_enabled, m1_aix_record_present, m2_aix_record_present, m1_total_macs, m2_total_macs,
                            dlc_one_base, dlc_two_base, csv_content)

        if copyrights:
            table_title('Models Copyright Differences', csv_content)
            if copyrights_same is False:
                print_copyrights_table(m1_copyright_str, m2_copyright_str, dlc_one_base, dlc_two_base, csv_content)
            else:
                print_value("No Differences found in dlcs copyright statements.\n", csv_content)

        if layers:
            table_title('Layer Differences', csv_content)
            # Check if models have unique layers
            if layers_same is False:
                print_layers_table(m1_unique_sorted, m2_unique_sorted, dlc_one_base, dlc_two_base, csv_content)
            else:
                print_value(layers_message, csv_content)

        if diff_by_id and parameters_same is None:
            # print layers causing 2 consecutive type difference. The common function for getting value differences
            # returns None if models are not comparable. So we can use any of the components to print out which layers
            # were the culprits
            print_value("Warning: Tool unable to diff by Layer id. If more than one consecutive layer type differences "
                        "is found, tool aborts. See below for details:", csv_content)
            print_value_diffs_for_layers(m1_diff_parameters, m2_diff_parameters, dlc_one_base, dlc_two_base,
                                         param_num_shared_layers, csv_content, "Two consecutive type",
                                         diff_strategy)
        else:
            if parameters:
                table_title('Parameter Differences', csv_content)
                # Check if shared layers have parameter differences
                if parameters_same is False:
                    print_value_diffs_for_layers(m1_diff_parameters, m2_diff_parameters, dlc_one_base, dlc_two_base,
                                                 param_num_shared_layers, csv_content, "Parameters",
                                                 diff_strategy)
                print_value(parameters_message, csv_content)

            if dimensions:
                table_title('Dimension Differences', csv_content)
                # Check if shared layers have dimension differences
                if dimensions_same is False:
                    print_value_diffs_for_layers(m1_diff_dimensions, m2_diff_dimensions, dlc_one_base, dlc_two_base,
                                                 dim_num_shared_layers, csv_content, "Dimensions",
                                                 diff_strategy)
                print_value(dimensions_message, csv_content)

            if outputs:
                table_title('Layer Output Name Differences', csv_content)
                # Check if corresponding layers output names are different
                if output_names_same is False:
                    print_value_diffs_for_layers(m1_diff_output_names, m2_diff_output_names, dlc_one_base, dlc_two_base,
                                                 output_names_num_shared_layers, csv_content, "Outputs",
                                                 diff_strategy)
                print_value(output_names_message, csv_content)

            if weights:
                table_title('Weight Differences', csv_content)
                # Check if corresponding layers have weight differences
                if weights_same is False:
                    print_value_diffs_for_layers(m1_diff_weights, m2_diff_weights, dlc_one_base, dlc_two_base,
                                                 weights_num_shared_layers, csv_content, "Weights",
                                                 diff_strategy)
                print_value(weights_message, csv_content)

            if output_tensor_types:
                table_title('Output Tensor Type Differences', csv_content)
                if output_tensor_types_same is False:
                    print_output_tensor_types_summary(tensor_comparison_data, dlc_one_base, dlc_two_base, csv_content)
                print_output_tensor_type_diffs(tensor_comparison_data, dlc_one_base, dlc_two_base, csv_content)

        if m1.is_aix_enabled():
            if hta:
                table_title('HTA Differences', csv_content)
                if aix_records_same is False:
                    print_aix_table(aix_record_keys, m1_aix_record_dict, m2_aix_record_dict, dlc_one_base, dlc_two_base,
                                    aix_error_messages, csv_content)
                elif not aix_error_messages:
                    # only print if no errors found. Since querying errors will lead to
                    # 2 models having same Aix records or no records in both
                    print_value(aix_records_message, csv_content)

                # print error message with loading aix records, if any
                for err_dict in aix_error_messages.values():
                    print_value(prRed(err_dict['msg']), csv_content)
                print('\n')
        copyrights_str = "--compare_copyrights"
        layers_str = "--compare_layers"
        parameters_str = "--compare_parameters"
        dimensions_str = "--compare_dimensions"
        weights_str = "--compare_weights"
        outputs_str = "--compare_outputs"
        output_tensor_types_str = "--compare_output_tensor_types"
        hta_str = "--compare_hta"
        # Outputs a help message for user based on boolean values from comparison functions
        print_value('', csv_content)
        if not copyrights:
            if copyrights_same is False:
                print_value(prRed('Info: Copyrights differences found. See ' + copyrights_str + ' for more details')+'\n', csv_content)
        if not layers:
            if layers_same is False:
                print_value(prRed('Info: Layer differences found. See ' + layers_str + ' for more details')+'\n', csv_content)
        if not parameters:
            if parameters_same is False:
                print_value(prRed('Info: Parameter differences found. See ' + parameters_str + ' for more details')+'\n', csv_content)
        if not dimensions:
            if dimensions_same is False:
                print_value(prRed('Info: Dimension differences found. See ' + dimensions_str + ' for more details')+'\n', csv_content)
        if not weights:
            if weights_same is False:
                print_value(prRed('Info: Weight differences found. See ' + weights_str + ' for more details')+'\n', csv_content)
        if not outputs:
            if output_names_same is False:
                print_value(prRed('Info: Output name differences found. See ' + outputs_str + ' for more details')+'\n', csv_content)
        if not output_tensor_types:
            if output_tensor_types_same is False:
                print_value(prRed('Info: Output Tensor Type differences found. See ' + output_tensor_types_str + ' for more details') + '\n', csv_content)

        if m1.is_aix_enabled():
            if not hta:
                if aix_error_messages:
                    print_value(prRed('Error: Querying HTA records returned error. See ' + hta_str + ' for more details')+'\n', csv_content)
                elif aix_records_same is False:
                    print_value(prRed('Info: HTA records differences found. See ' + hta_str + ' for more details')+'\n', csv_content)
        pass
