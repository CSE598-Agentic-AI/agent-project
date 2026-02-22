import os
import json


class ProgressViewer():

    def __init__(self, model_size, env, strategy, folder_path = None, file_path = None):
        self.folder_path = folder_path
        self.model_size = model_size
        self.env = env
        self.strategy = strategy
        self.file_path = file_path

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path
    
    def set_file_path(self, file_path):
        self.file_path = file_path


    

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


    def missing_task_ids(self):
        if not os.path.exists(self.file_path):
            return None
        with open(self.file_path, "r") as f:
            data = json.load(f)
            completed = set([item["task_id"] for item in data])
        path_parts = self.file_path.replace("\\", "/").split("/")
        env = path_parts[1] if len(path_parts) > 1 else "airline"
        agent = path_parts[2] if len(path_parts) > 2 else "fc"
        model = path_parts[3] if len(path_parts) > 3 else "4B"
        trial_str = (path_parts[-1] if path_parts else "num_trials-1.json").split("-")[-1].split(".")[0]
        try:
            trial = int(trial_str)
        except ValueError:
            trial = 1
        total_tasks = 115 if env == "retail" else 50
        missing = [i for i in range(total_tasks) if i not in completed]
        if len(missing) == 0:
            print(f"\n{self.file_path}")
            print(f"  All tasks have at least one trial completed. Check task_id to trial mapping above to see if any trials are missing.")
            return None
        completed_count = len(completed)
        missing_str = ", ".join(map(str, missing))
        print(f"\n{self.file_path}")
        print(f"  {completed_count}/{total_tasks} completed | missing count: {len(missing)} | break down: {missing_str}")
        # print(f"  missing: {missing_str}")

        # Compute sbatch job index (from README mapping) and print the exact command to run
        env_base = 0 if env == "airline" else 60  # airline: 0-59, retail: 60-119
        agent_offset_map = {"act": 0, "react": 20, "fc": 40}
        model_offset_map = {"4B": 0, "8B": 5, "14B": 10, "32B": 15}
        agent_offset = agent_offset_map.get(agent)
        model_offset = model_offset_map.get(model)
        if agent_offset is not None and model_offset is not None and 1 <= trial <= 5:
            job_index = env_base + agent_offset + model_offset + (trial - 1)
            assistant_model = f"Qwen/Qwen3-{model}-Instruct-2507"
            cmd = f"sbatch tau-experiment.sh {env} {agent} {assistant_model} {trial}  # {job_index}"
            print(f"  cmd: {cmd}")


    def count_completed_tasks_in_folder(self, total_tasks):
        """
        Count how many completed tasks there are in all .json files in a directory,
        and print the percentage completion for each file.
        """
        print(f"Counting completed tasks in {self.folder_path}")

        for i in range(0, 5):
            file = f"num_trials-{i + 1}.json"
            file_path = os.path.join(self.folder_path, file)
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Could not load {file}: {e}")
                    continue

            completed_tasks = [item for item in data if not item.get("info", {}).get("error")]
            num_completed = len(completed_tasks)
            
            percent_complete = 100.0 * num_completed / (total_tasks * (i + 1))
            print(f"{file}: {num_completed}/{(total_tasks * (i + 1))} tasks completed ({percent_complete:.2f}%)")
        print()
        return num_completed

    def progress_by_model(self):
        trials = [1, 2, 3, 4, 5]

        retail_tasks = 115 
        airline_tasks = 50 
        total_retail_tasks = 0
        total_airline_tasks = 0
        for trial in trials:
            total_retail_tasks += retail_tasks * trial
            total_airline_tasks += airline_tasks * trial
        total_tasks = total_retail_tasks + total_airline_tasks
        completion = self.count_completed_tasks_in_folder(retail_tasks)
        completion += self.count_completed_tasks_in_folder(retail_tasks)
        completion += self.count_completed_tasks_in_folder(retail_tasks)
        
        completion += self.count_completed_tasks_in_folder(airline_tasks)
        completion += self.count_completed_tasks_in_folder(airline_tasks)
        completion += self.count_completed_tasks_in_folder(airline_tasks)
        completion = completion / total_tasks * 100 # 3 strategies (act, react, fc)
        print(f"Qwen{self.model_size} completion: {completion:.2f}%")


    def detailed_progress(self):
        print(f"Printing task trials in {self.folder_path}      -------------------------")
        self.run_on_all_files_in_folder(self.print_task_trials(group_by_task=True))
        print(f"Printing missing task ids in {self.folder_path} -------------------------")
        self.run_on_all_files_in_folder(self.missing_task_ids)

