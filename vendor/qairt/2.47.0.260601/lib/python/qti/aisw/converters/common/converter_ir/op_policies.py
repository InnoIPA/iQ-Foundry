# ==============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================


class ConversionNamePolicy(object):
    def __init__(self):
        self.type_count = {}
        self.exist_names = set()

    def reserve_op_names(self, names):
        self.exist_names.update(names)

    def get_unique_name(self, name_prefix_str, op_type):
        count = self.type_count.get(op_type, 0)
        while True:
            candidate_name = "%s_%d" % (name_prefix_str, count)
            if candidate_name not in self.exist_names:
                break
            count += 1
        self.type_count[op_type] = count
        self.exist_names.add(candidate_name)
        return candidate_name

    def get_op_name(self, op):
        if hasattr(op, 'LEGACY_TRANSLATION_KEY'):
            name_prefix_str = str(op.LEGACY_TRANSLATION_KEY)
        else:
            name_prefix_str = str(op.type)
        if op.name:
            self.exist_names.add(str(op.name))
            return str(op.name)
        else:
            return self.get_unique_name(name_prefix_str, op.type)

    def get_op_name_by_type(self, op_type, legacy_translation_key, custom_op_type="", **kwargs):
        if legacy_translation_key:
            name_prefix_str = str(legacy_translation_key)
        else:
            name_prefix_str = str(op_type)

        #if it is a folded op then we add _ at start to distinguish with other ops
        if 'folded_op' in kwargs:
            if kwargs['folded_op']:
                name_prefix_str = '_' + name_prefix_str

        if "common_prefix" in kwargs:
            if len(kwargs['common_prefix']) and kwargs["common_prefix"] != '/':
                final_name = "%s_%s" % (name_prefix_str, kwargs['common_prefix'])
            else:
                final_name = "%s_%s" % (name_prefix_str, kwargs['out_buf_name'])
            self.exist_names.add(final_name)
            return final_name

        else:
            return self.get_unique_name(name_prefix_str, op_type)

    def get_input_names(self, op, input_names):
        return list(map(str, input_names))

    def get_output_names(self, op, output_names):
        return list(map(str, output_names))

    def remove_output_name(self, output_name):
        return


class ConversionShapeInferencePolicy(object):

    def infer_shape(self, op, input_shapes):
        raise NotImplementedError("infer_shape for {} not implemented ".format(str(self.__class__.__name__)))
