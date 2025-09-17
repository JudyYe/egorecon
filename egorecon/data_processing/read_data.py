#!/usr/bin/env python3
import argparse
import pickle
import sys

def load_pickle(path):
    """
    Load and return the object stored in a pickle file.
    Raises an exception if loading fails.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(
        description="Load and display the contents of a pickle (.pkl) file."
    )
    parser.add_argument(
        "pkl_file",
        help="Path to the pickle file to read (e.g. data.pkl)"
    )
    parser.add_argument(
        "--max-items", "-n",
        type=int,
        default=3,
        help="If the loaded object is a list/tuple/dict, print at most this many items."
    )
    args = parser.parse_args()

    try:
        data = load_pickle(args.pkl_file)
    except FileNotFoundError:
        print(f"Error: File not found: {args.pkl_file}", file=sys.stderr)
        sys.exit(1)
    except (pickle.UnpicklingError, EOFError) as e:
        print(f"Error: Failed to unpickle file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

    # Nicely display the loaded object
    if isinstance(data, (list, tuple)):
        print(f"Loaded {type(data).__name__} of length {len(data)}:")
        # for i, item in enumerate(data[: args.max_items]):
        #     print(f"  [{i}]: {item!r}")
        # if len(data) > args.max_items:
        #     print(f"  ... and {len(data) - args.max_items} more items.")
    elif isinstance(data, dict):
        print(f"Loaded dict with {len(data)} keys:")
        for i, (k, v) in enumerate(data.items()):
            if i >= args.max_items:
                print(f"  ... and {len(data) - args.max_items} more items.")
                break
            print(f"  {k!r}")

            for ii, (kk, vv) in enumerate(v.items()):
                if ii >= args.max_items:
                    print(f"  ... and {len(v) - args.max_items} more items.")
                    break
                print(f"  {kk!r}: {len(vv)!r}")

                for iii, vvv in enumerate(vv):
                    if iii >= args.max_items:
                        print(f"  ... and {len(vv) - args.max_items} more items.")
                        break
                    print(f"  {iii!r}: {vvv!r}")
    else:
        # Fallback for any other object type
        print(f"Loaded object of type {type(data).__name__}:")
        print(data)

if __name__ == "__main__":
    main()
