import pickle

def append_list_of_dicts_to_pickle(data, filename):
    """Appends a list of dictionaries to a pickle file."""
    try:
        with open(filename, "ab") as f:
            pickle.dump(data, f)
    except FileNotFoundError:
        with open(filename, "wb") as f:
            pickle.dump(data, f)

def read_list_of_dicts_from_appended_pickle(filename):
    """Reads lists of dictionaries from a pickle file that was appended to."""
    results = []
    try:
        with open(filename, "rb") as f:
            while True:
                results.extend(pickle.load(f))  # Extend to add all dictionaries
    except EOFError:
        pass
    return results