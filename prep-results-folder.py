import os
import shutil


def migrate_local(root_results: str):
    """
    For every `local` folder under `root_results`:
      1. Delete any `num_trials-*.json` files already inside the `local` folder.
      2. Move all `num_trials-*.json` files from the parent folder into the `local` folder.
    """
    if not os.path.isdir(root_results):
        return

    # Walk the tree so we handle every `{...}/local` directory beneath `root_results`
    files = os.listdir(root_results)
    print(f"Files: {files}")
    print(f"Root: {root_results}")
    
    if "local" not in os.listdir(root_results):
        return

    local_dir = os.path.join(root_results, "local")
    print(f"Migrating {local_dir}")
    # 1) Remove any existing num_trials-*.json files in the local folder
    for name in os.listdir(local_dir):
        if name.startswith("num_trials-") and name.endswith(".json"):
            try:
                os.remove(os.path.join(local_dir, name))
                print(f"Removing {os.path.join(local_dir, name)}")
            except OSError:
                # Ignore files we fail to remove; continue with the rest
                pass

    # 2) Move num_trials-*.json from the parent folder into the local folder
    for name in files:
        if name.startswith("num_trials-") and name.endswith(".json"):
            src = os.path.join(root_results, name)
            dst = os.path.join(local_dir, name)
            try:
                os.replace(src, dst)
                print(f"Moving {src} to {dst}")
            except OSError:
                # If move fails, skip this file and continue
                pass 
    print()       


def migrate_new(root_results: str, results_new: str):
    """
    For every `new` folder under `root_results`:
      1. Delete any `num_trials-*.json` files already inside the `new` folder.
      2. Move all `num_trials-*.json` files from the parent folder into the `new` folder.
    """
    if not os.path.isdir(results_new):
        return
    
    files = os.listdir(results_new)
    print(f"Files: {files}")
    print(f"Results New: {results_new}")
    print(f"Root Results: {root_results}")

    print(f"Migrating {results_new} to {root_results}")

    for name in files:
        if name.startswith("num_trials-") and name.endswith(".json"):
            src = os.path.join(results_new, name)
            dst = os.path.join(root_results, name)
            try:
                os.replace(src, dst)
                print(f"Moving {src} to {dst}")
            except OSError:
                # If move fails, skip this file and continue
                pass 
    print() 
    print()      

if __name__ == "__main__":
    # Match the root style: results/{env}/{strategy}/{model_size}
    
    root_results = "results"
    new_results = "results-new"
    ENVS = ["airline", "retail"]
    STRATEGIES = ["act", "react", "fc"]
    MODEL_SIZES = ["4B", "8B", "14B", "32B"]
    for env in ENVS:
        for strategy in STRATEGIES:
            for model_size in MODEL_SIZES[:1]:
                folder_path = os.path.join(root_results, env, strategy, model_size)
                new_folder_path = os.path.join(new_results, env, strategy, model_size)
                migrate_local(folder_path)
                migrate_new(folder_path, new_folder_path)