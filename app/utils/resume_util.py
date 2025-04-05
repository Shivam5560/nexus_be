import random
import string
import re
import os
def generate_unique_id(length=8):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length)).upper()


def check_unique_file_name(newfilename, files):
    """
    Checks for filename conflicts and generates a unique filename if needed.

    Args:
        newfilename: The base filename (without extension) of the file to be uploaded.
        files: A list of file dictionaries, where each dictionary should have a 'file_name' key.

    Returns:
        tuple: (unique_filename, status_code)
            - unique_filename: The unique filename (with extension) to use.
            - status_code: 200 (unique), or 409 (conflict/renamed).
    """

    new_name, new_ext = os.path.splitext(newfilename)
    max_suffix = 0
    conflicts = []

    for file_dict in files:
        try:
            name, ext = os.path.splitext(file_dict['file_name'])
            # print(name, ext, new_name, new_ext)
            if name == new_name and ext == new_ext:
                conflicts.append(file_dict['file_name'])
                match = re.search(rf"{re.escape(name)}-(\d+){re.escape(ext)}$", file_dict['file_name'])
                print(file_dict['file_name']," ",newfilename)
                if match:
                    max_suffix = max(max_suffix, int(match.group(1)))

        except KeyError:
            print(f"Warning: Dictionary missing 'file_name' key: {file_dict}")
            continue
        except TypeError:
            print(f"Warning: file_dict['file_name'] is not a string: {file_dict}")
            continue

    if conflicts:
        unique_filename = f"{new_name}-{max_suffix + 1}{new_ext}"
        return unique_filename, 409  # Conflict, renamed
    else:
        return newfilename, 200  # Unique