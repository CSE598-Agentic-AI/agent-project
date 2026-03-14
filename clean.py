import json
import os
import re
from typing import List


class Cleaner():

    def __init__(self, model_size, env, strategy, folder_path = None, file_path = None):
        self.model_size = model_size
        self.env = env
        self.strategy = strategy
        self.folder_path = folder_path
        self.file_path = file_path

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path
        
    def set_file_path(self, file_path):
        self.file_path = file_path

    def remove_error_logs(self) -> None:
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Remove any tasks that have an error recorded
        cleaned_data = [
            item
            for item in data
            if not (item.get("info", {}).get("error") is not None)
        ]

        with open(self.file_path, "w") as f:
            json.dump(cleaned_data, f, indent=4)
            
    def error_task_ids(self) -> None:
        if not os.path.exists(file_path):
            return
        with open(file_path, "r") as f:
            data = json.load(f)
            
        return [item["task_id"] for item in data if item.get("info", {}).get("error") is not None]
            

    def sort_by_task_id(self) -> None:
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r") as f:
            data = json.load(f)

        data.sort(key=lambda x: (x["task_id"], x.get("trial", 0)))

        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=4)
            
    def remove_duplicate_tasks(self):
        """
        Resolve duplicate (task_id, trial) pairs by renumbering trials and
        trimming so each task_id has exactly num_trials trials (0, 1, ..., num_trials-1).

        - For a file named num_trials-4.json, each task_id will end up with
          exactly trials 0, 1, 2, 3. Extra trials (e.g. trial 4, 5, ...) are deleted.
        - Duplicate trial values (e.g. two records with trial 0) are renamed to
          new consecutive indices: within each task, indices are sorted by
          (original trial, list position), then assigned 0, 1, 2, ...
        - If the filename does not match num_trials-N.json, no trimming is done.

        Intended to be used with run_on_all_files_in_folder, e.g.:

            cleaner = Cleaner(model_size, env, strategy, folder_path=...)
            cleaner.run_on_all_files_in_folder(cleaner.remove_duplicate_tasks)
        """
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            return f"Error reading {self.file_path}: {e}"

        if not isinstance(data, list):
            return f"Skipping {self.file_path}: JSON root is not a list"

        # Parse expected number of trials from filename (e.g. num_trials-4.json -> 4)
        basename = os.path.basename(self.file_path)
        num_trials_match = re.match(r"num_trials-(\d+)\.json$", basename)
        num_trials = int(num_trials_match.group(1)) if num_trials_match else None

        # Group indices by task_id
        task_id_to_indices = {}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            task_id = item.get("task_id")
            task_id_to_indices.setdefault(task_id, []).append(i)

        # For each task, sort indices by (original trial, index) so we keep first num_trials
        # by trial order; duplicates (same trial) get consecutive new indices 0, 1, 2, ...
        index_to_new_trial = {}
        indices_to_keep = set()
        for task_id, indices in task_id_to_indices.items():
            keep_count = min(len(indices), num_trials) if num_trials is not None else len(indices)
            # Sort by original trial then by list position so ordering is deterministic
            sorted_indices = sorted(
                indices,
                key=lambda idx: (data[idx].get("trial", 0), idx),
            )
            for new_trial, idx in enumerate(sorted_indices[:keep_count]):
                indices_to_keep.add(idx)
                index_to_new_trial[idx] = new_trial

        # Build new data: keep only allowed indices, in original list order, with trial renumbered
        new_data = []
        renumber_count = 0
        delete_count = len(data) - len(indices_to_keep)

        for i, item in enumerate(data):
            if i not in indices_to_keep:
                continue
            item = dict(item)
            new_trial = index_to_new_trial[i]
            if item.get("trial") != new_trial:
                item["trial"] = new_trial
                renumber_count += 1
            new_data.append(item)

        if renumber_count == 0 and delete_count == 0:
            return None

        try:
            with open(self.file_path, "w") as f:
                json.dump(new_data, f, indent=4)
        except Exception as e:
            return f"Error writing {self.file_path}: {e}"

        msg = f"{self.file_path}:"
        if delete_count > 0:
            msg += f" removed {delete_count} extra trial(s);"
        if renumber_count > 0:
            msg += f" renumbered {renumber_count} trial(s)"
        return msg.rstrip(";").strip()
            
            
    def show_trials_done(file_path: str) -> None:
        """
        Shows which trials for each task in the file are done.

        Prints the mapping: task_id -> [list of trial indices present in the file]
        """
        if not os.path.exists(self.file_path):
            print(f"{self.file_path} does not exist.")
            return

        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Map task_id -> set of trial numbers observed
        # If 'trial' field not present, just list each task_id occurrence.
        task_trials = {}
        for item in data:
            task_id = item.get("task_id")
            trial_num = item.get("trial", None)
            if task_id is not None:
                if task_id not in task_trials:
                    task_trials[task_id] = set()
                if trial_num is not None:
                    task_trials[task_id].add(trial_num)
                else:
                    # No trial field; just use occurrence count
                    task_trials[task_id].add("<present>")

        print(f"\nTrials recorded per task_id in {self.file_path}:")
        for task_id in sorted(task_trials.keys()):
            trials = sorted(
                list(task_trials[task_id]),
                key=lambda x: (isinstance(x, int), x)
            )
            print(f"  task_id {task_id}: trials = {trials}")
        
            
    def show_jobs_to_run(rerun_job_indices):
        if rerun_job_indices:
            unique_indices = sorted(set(rerun_job_indices))
            ranges = []
            start = None
            prev = None
            for idx in unique_indices:
                if start is None:
                    start = prev = idx
                elif idx == prev + 1:
                    prev = idx
                else:
                    ranges.append((start, prev))
                    start = prev = idx
            if start is not None:
                ranges.append((start, prev))

            print("\nConsolidated sbatch array suggestions:")
            for a, b in ranges:
                if a == b:
                    print(f"  sbatch --array={a} tau-experiment.sh")
                else:
                    print(f"  sbatch --array={a}-{b} tau-experiment.sh")
                    
            
    def run_on_all_files(function):
        base_path = "ben"
        environments = ["airline", "retail"]
        agents = ["act", "react", "fc"]
        models = ["4B", "8B", "14B", "32B"]
        trials = [1, 2, 3, 4, 5]
        rerun_job_indices = []
        for env in environments:
            for agent in agents:
                for model in models:
                    for trial in trials:
                        file = f"{base_path}/{env}/{agent}/{model}/num_trials-{trial}.json"
                        function(file)

                        
    def run_on_all_files_in_folder(self, function):
        for file in os.listdir(self.folder_path):
            if file.endswith(".json"):
                self.set_file_path(os.path.join(self.folder_path, file))
                result = function()
                if result is not None:  
                    print(result)
                    
                    
    def print_task_trials(self, group_by_task=False):
        """
        For the given file path, print each (task_id, trial) observed in the data.
        If no 'trial' field is present for a task, prints 'None' as its trial.

        If group_by_task is True, prints a hash-map style summary:
            task_id -> [sorted unique list of trials observed]
        """
        import json
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            return

        if not group_by_task:
            print(f"\nTask IDs and trial numbers in {self.file_path}:")
            for item in data:
                task_id = item.get("task_id")
                trial_num = item.get("trial", None)
                print(f"  task_id: {task_id}, trial: {trial_num}")
            return

        # Group/hash-map style: task_id -> list of trials
        task_to_trials = {}
        for item in data:
            task_id = item.get("task_id")
            trial_num = item.get("trial", None)
            if task_id is None:
                continue
            if task_id not in task_to_trials:
                task_to_trials[task_id] = []
            task_to_trials[task_id].append(trial_num)

        print(f"\nTask IDs mapped to trials in {self.file_path}:")
        for task_id in sorted(task_to_trials.keys()):
            observed = task_to_trials[task_id]
            has_none = any(t is None for t in observed)
            uniques = sorted({t for t in observed if t is not None})
            if has_none:
                trials_list = [None] + uniques
            else:
                trials_list = uniques
            print(f"  {task_id}: {trials_list}")
        return task_to_trials
                    
                    
    def task_differences(self, file_name):
        local_file_path = os.path.join(self.folder_path, "local", file_name)
        main_file_path = os.path.join(self.folder_path, file_name)
        local_task_to_trials = self.print_task_trials(group_by_task=True)
        main_task_to_trials = self.print_task_trials(group_by_task=True)

        # If either file failed to load, skip processing to avoid NoneType iteration
        if not isinstance(local_task_to_trials, dict) or not isinstance(main_task_to_trials, dict):
            print(f"Skipping differences: missing or unreadable files:")
            print(f"  local: {self.local_file_path}")
            print(f"  main:  {self.main_file_path}")
            return

        old_to_new = []
        for task_id in local_task_to_trials:
            if task_id not in main_task_to_trials:
                print(f"Task {task_id} is missing from newest file\n")
                old_to_new.append(task_id)
            else:
                if local_task_to_trials[task_id] != main_task_to_trials[task_id]:
                    print(f"Task {task_id} in the newest file needs an update")
                    for trial in local_task_to_trials[task_id]:
                        if trial not in main_task_to_trials[task_id]:
                            print(f"Trial {trial} is missing from newest file\n")
                            
        with open(local_file_path, "r") as f:
            local_data = json.load(f)
        with open(main_file_path, "r") as f:
            main_data = json.load(f)

        # Append every item from local whose task_id is in old_to_new into the new (main) file
        import copy
        old_to_new_set = set(old_to_new)
        added_count = 0
        for item in local_data:
            if item.get("task_id") in old_to_new_set:
                main_data.append(copy.deepcopy(item))
                added_count += 1

        with open(main_file_path, "w") as f:
            json.dump(main_data, f, indent=4)
        print(f"Appended {added_count} item(s) for {len(old_to_new_set)} task(s) from {local_file_path} to {main_file_path}")


    
    

    
    