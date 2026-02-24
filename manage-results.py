"""
Main entry point for cleaning experiment results and viewing progress.

This script:
  1. Sorts JSON result files by task_id and trial. Sometimes the files are not sorted correctly because of all the stopping and starting of the experiments.
  2. Removes failed/errored task entries from the files
  3. Prints an aggregate completion summary across all envs/strategies for the model

All operations target the results/ folder structure: results/{env}/{strategy}/{model_size}/
"""

import os
from progress import ProgressViewer
from clean import Cleaner
from evaluate import Evaluator


if __name__ == "__main__":
  # -------------------------------------------------------------------------
  # CONFIGURATION - Edit these to target a specific experiment subset
  # -------------------------------------------------------------------------
  model_size = "4B"   # Options: "4B", "8B", "14B", "32B"
  env = "airline"     # Options: "airline" (50 tasks), "retail" (115 tasks)
  strategy = "react"    # Options: "act", "react", "fc"

  folder_path = f"results/{env}/{strategy}/{model_size}"

  progress_viewer = ProgressViewer(model_size, env, strategy, folder_path)
  cleaner = Cleaner(model_size, env, strategy, folder_path)
  evaluator = Evaluator(model_size, env, strategy, folder_path)

  # -------------------------------------------------------------------------
  # STEP 1: Clean and sort the JSON files in the target folder
  # -------------------------------------------------------------------------
  # Sorts entries by (task_id, trial) for consistent ordering
  print(f"Sorting files in {folder_path}            --------------------------")
  cleaner.run_on_all_files_in_folder(cleaner.sort_by_task_id)

  # Removes tasks that have error logs (failed runs) from the JSON files
  print(f"Removing error logs from {folder_path}    --------------------------")
  cleaner.run_on_all_files_in_folder(cleaner.remove_error_logs)
  print()

  # -------------------------------------------------------------------------
  # STEP 2: View progress
  # -------------------------------------------------------------------------
  # progress_by_model() prints completion % for ALL strategies (act, react, fc)
  # across BOTH envs (retail, airline) for this model_size. Expect output like:
  #   "Counting completed tasks in results/retail/act/4B"
  #   "num_trials-1.json: X/Y tasks completed (Z%)"
  #   ... (repeated for each strategy and env)
  #   "Qwen4B completion: XX.XX%"
  progress_viewer.progress_by_model()

  # detailed_progress() gives a per-folder view: task_id -> trials mapping and
  # missing task IDs (useful for resuming failed experiments)
  progress_viewer.detailed_progress(folder_path)

  # -------------------------------------------------------------------------
  # STEP 3: Evaluate the results and graph progress
  # -------------------------------------------------------------------------
  # For each num_trials-*.json file in the folder:
  #   - Reads task_id, trial, and score (reward/score/result) from each entry
  #   - Prints a table of (task_id, trial, score)
  #   - Pivots data into a heatmap: rows=task_id, columns=trial, color=score
  #   - Shows a matplotlib window with viridis colormap (darker=lower, lighter=higher)
  #   - plt.show() blocks until you close each figure; close to see the next file
  # print(f"Evaluating results in {folder_path}            --------------------------")
  # evaluator.evaluate_folder()


