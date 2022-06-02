from typing import Any, Dict


def update_config_from_string(config: Dict[str, Any], update_str: str):
    """
    Updates values for keys of the `config` dict based on `update_str`.

    The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
    "num_beams=1,no_repeat_ngram_size=3"

    The keys to change have to already exist in the config dict.

    Args:
        update_str (`str`): String with key-values that should be updated for this class.
    """
    # inspired by https://github.com/huggingface/transformers/blob/216499bfcc6933946bd565911dd570f2a1f174c0/src/transformers/configuration_utils.py#L836
    d = dict(x.split("=") for x in update_str.split(","))
    for k, v in d.items():
        if k not in config:
            raise ValueError(f"key {k} isn't in the original config dict")
        old_v = config[k]
        if isinstance(old_v, bool):
            if v.lower() in ["true", "1", "y", "yes"]:
                v = True
            elif v.lower() in ["false", "0", "n", "no"]:
                v = False
            else:
                raise ValueError(f"can't derive true or false from {v} (key {k})")
        elif isinstance(old_v, int):
            v = int(v)
        elif isinstance(old_v, float):
            v = float(v)
        elif not isinstance(old_v, str):
            raise ValueError(
                f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
            )
        config[k] = v
    return config
