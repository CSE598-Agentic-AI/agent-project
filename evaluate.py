
import os
import json
import matplotlib.pyplot as plt
import pandas as pd

class Evaluator():
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

    def evaluate_file(self):
        """
        Reads a file, extracts result for every task_id/trial, prints table, and plots the results.
        Assumes each item in data has fields: 'task_id', 'trial', and a score/result field (e.g., 'reward' or 'score').
        """
        # Read data
        with open(self.file_path, "r") as f:
            data = json.load(f)

        # Extract relevant fields for table and plot
        records = []
        for item in data:
            task_id = item.get('task_id')
            trial = item.get('trial')
            # Try the most common result fields, default to None if not present
            score = item.get('reward', item.get('score', item.get('result', None)))
            records.append({"task_id": task_id, "trial": trial, "score": score})

        # Create DataFrame for table
        df = pd.DataFrame(records)
        df = df.sort_values(by=["task_id", "trial"])
        print(f"{self.file_path}")
        print(df.to_string(index=False))

        # Pivot for heatmap: Rows=task_id, Cols=trial, Values=score
        print("df: \n", df)


        pivot = df.pivot(index='task_id', columns='trial', values='score')
        plt.figure(figsize=(10, 6))
        plt.title(f"{self.file_path}")
        ax = plt.gca()
        cax = ax.matshow(pivot, cmap="viridis", aspect="auto")
        plt.xlabel("Trial")
        plt.ylabel("Task ID")
        plt.colorbar(cax, label="Score")
        plt.xticks(
            range(len(pivot.columns)), 
            pivot.columns if pivot.columns.is_monotonic_increasing else range(len(pivot.columns))
        )
        plt.yticks(
            range(len(pivot.index)), 
            pivot.index if pivot.index.is_monotonic_increasing else range(len(pivot.index))
        )
        plt.tight_layout()
        plt.show()
        return df


    def evaluate_folder(self):
        for file in os.listdir(self.folder_path):
            if file.endswith(".json"):
                self.set_file_path(os.path.join(self.folder_path, file))
                result = self.evaluate_file()
                if result is not None:  
                    print(result)


