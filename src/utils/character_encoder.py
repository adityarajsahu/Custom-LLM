import json
import os

def string_to_integer(list_of_characters):
    str_to_int_map = {}
    for i, char in enumerate(list_of_characters):
        str_to_int_map[char] = i

    json_dir_path = os.path.join(os.curdir, "src/maps")
    if not os.path.isdir(json_dir_path):
        os.mkdir(json_dir_path)
    json_file_path = os.path.join(json_dir_path, "str_to_int.json")
    with open(json_file_path, "w") as file:
        json.dump(str_to_int_map, file, indent = 4)

    return str_to_int_map

def integer_to_string(list_of_characters):
    int_to_str_map = {}
    for i, char in enumerate(list_of_characters):
        int_to_str_map[i] = char
    
    json_dir_path = os.path.join(os.curdir, "src/maps")
    if not os.path.isdir(json_dir_path):
        os.mkdir(json_dir_path)
    json_file_path = os.path.join(json_dir_path, "int_to_str.json")
    with open(json_file_path, "w") as file:
        json.dump(int_to_str_map, file, indent = 4)
    
    return int_to_str_map

def encode(input_string):
    json_file_path = os.path.join(os.curdir, "src/maps/str_to_int.json")
    with open(json_file_path, "r") as file:
        str_to_int_map = json.load(file)

    encoding_list = []
    for char in input_string:
        encoded_val = str_to_int_map[char]
        encoding_list.append(encoded_val)

    return encoding_list

def decode(encoding_list):
    json_file_path = os.path.join(os.curdir, "src/maps/int_to_str.json")
    with open(json_file_path, "r") as file:
        int_to_str_map = json.load(file)

    decoded_string = ""
    for encoded_val in encoding_list:
        decoded_char = int_to_str_map[encoded_val]
        decoded_string += decoded_char

    return decoded_string