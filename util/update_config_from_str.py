from typing import Any, Dict


def update_config_from_string(config: Dict[str, Any], update_str: str):
    """
    Updates values for keys of the `config` dict based on `update_str`.

    The expected format for `update_str` is ints, floats and strings as is,
    for booleans use `true` or `false`, and for None use `null` or `none`.
    For example:"num_beams=1;no_repeat_ngram_size=3;length_penalty=none".
    """
    if update_str == "":
        return config
    d = dict(x.split("=") for x in update_str.split(";"))
    for k, v in d.items():
        if v.lower() in ["null", "none"]:
            v = None
        elif v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        else:
            try:
                v = eval(v)
            except Exception:
                pass  # not a float or int, leave as is
        config[k] = v
    return config
