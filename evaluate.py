
import os
import json
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_file(file_path):
    """
    Reads a file, extracts result for every task_id/trial, prints table, and plots the results.
    Assumes each item in data has fields: 'task_id', 'trial', and a score/result field (e.g., 'reward' or 'score').
    """
    # Read data
    with open(file_path, "r") as f:
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
    print("Task Results Table:")
    print(df.to_string(index=False))

    # Pivot for heatmap: Rows=task_id, Cols=trial, Values=score
    print("df: \n", df)


    pivot = df.pivot(index='task_id', columns='trial', values='score')
    plt.figure(figsize=(10, 6))
    plt.title("Score by Task ID and Trial")
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


def run_on_all_files_in_folder(folder_path, function):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            result = function(os.path.join(folder_path, file))
            if result is not None:  
                print(result)




if __name__ == "__main__":
    
    model_size = "4B" # 4B, 8B, 14B, 32B
    env = "airline" # retail, airline
    strategy = "react" # act, react, fc
    folder_path = f"results/{env}/{strategy}/{model_size}"

    
    
    print(f"Sorting files in {folder_path}            --------------------------")
    run_on_all_files_in_folder(folder_path, evaluate_file)