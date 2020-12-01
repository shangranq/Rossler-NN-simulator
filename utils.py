import json

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def cast_to_float(X_i, Y_i, Z_i, X_o, Y_o, Z_o):
    X_i = X_i.float()
    Y_i = Y_i.float()
    Z_i = Z_i.float()
    X_o = X_o.float()
    Y_o = Y_o.float()
    Z_o = Z_o.float()
    return X_i, Y_i, Z_i, X_o, Y_o, Z_o