import json
import _jsonnet


class ConfigReader:

    def __init__(self, config_file, ext_vars=None):
        self.config = json.loads(_jsonnet.evaluate_file(config_file, ext_vars=ext_vars))

    def read(self, cls):
        init_params = {}
        stack = [('', self.config)]

        while stack:
            prefix, value = stack.pop()

            if isinstance(value, dict):
                for key, sub_value in value.items():
                    if type(sub_value) == str:
                        if sub_value == 'true':
                            sub_value = True
                        elif sub_value == 'false':
                            sub_value = False
                        elif sub_value.replace('-', '').isnumeric():
                            sub_value = int(sub_value)
                        elif '.' in sub_value and sub_value.replace('-', '').replace('.', '').isnumeric():
                            sub_value = float(sub_value)

                    stack.append((f"{prefix}{key}__", sub_value))
            else:
                param_name = prefix[:-2]
                init_params[param_name] = value

        init_params['trainer__config'] = self.config
        return cls(**init_params)
