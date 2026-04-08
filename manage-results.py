import os
from progress import ProgressViewer
from clean import Cleaner
from evaluate import Evaluator


if __name__ == "__main__":
  # -------------------------------------------------------------------------
  # CONFIGURATION - All folder paths: results/{env}/{strategy}/{model_size}
  # -------------------------------------------------------------------------
  ENVS = ["airline", "retail"]
  STRATEGIES = ["act", "react", "fc"]
  MODEL_SIZES = ["4B", "8B", "14B", "32B"]
  ALL_FILES = True
  CLEAN = False
  EVALUATE = False
  
  if ALL_FILES:
    for env in ENVS:
      for strategy in STRATEGIES:
        for model_size in MODEL_SIZES[:1]: # Currently only running 4B (Ben)
          folder_path = os.path.join("results", env, strategy, model_size)
          if not os.path.isdir(folder_path):
            continue

          progress_viewer = ProgressViewer(model_size, env, strategy, folder_path)
          cleaner = Cleaner(model_size, env, strategy, folder_path)
          evaluator = Evaluator(model_size, env, strategy, folder_path)
          
          # -------------------------------------------------------------------------
          # STEP 1: Clean and sort the JSON files in every target folder
          # -------------------------------------------------------------------------
          if CLEAN:
            # Removes tasks that have error logs (failed runs) from the JSON files
            print(f"Removing error logs from {folder_path}    --------------------------")
            cleaner.run_on_all_files_in_folder(cleaner.remove_error_logs)
            
            # Removes duplicate tasks from the JSON files
            print(f"Removing duplicate tasks from {folder_path}    --------------------------")
            cleaner.run_on_all_files_in_folder(cleaner.remove_duplicate_tasks)

            # Sorts entries by (task_id, trial) for consistent ordering
            print(f"Sorting files in {folder_path}            --------------------------")
            cleaner.run_on_all_files_in_folder(cleaner.sort_by_task_id)
            print()
          
          # Evaluates the results and graphs the progress
          if EVALUATE:
            print(f"Evaluating results in {folder_path}            --------------------------")
            evaluator.evaluate_folder()
            print()

      
    
    # -------------------------------------------------------------------------
    # STEP 2: View progress
    # -------------------------------------------------------------------------
    # progress_by_model() prints completion % for ALL strategies (act, react, fc)
    # across BOTH envs (retail, airline) for this model_size.
    for model_size in MODEL_SIZES[:1]: # Currently only running 4B 
      progress_viewer = ProgressViewer(model_size, ENVS[0], STRATEGIES[0], None)
      progress_viewer.progress_by_model()
      

  else:
    model_size = "4B"
    env = "airline"
    strategy = "act"
    folder_path = os.path.join("results", env, strategy, model_size)
    progress_viewer = ProgressViewer(model_size, env, strategy, folder_path)
    cleaner = Cleaner(model_size, env, strategy, folder_path)
    evaluator = Evaluator(model_size, env, strategy, folder_path)
    
    if CLEAN:
      print(f"Removing error logs from {folder_path}    --------------------------")
      cleaner.run_on_all_files_in_folder(cleaner.remove_error_logs)
      
      # Removes duplicate tasks from the JSON files
      print(f"Removing duplicate tasks from {folder_path}    --------------------------")
      cleaner.run_on_all_files_in_folder(cleaner.remove_duplicate_tasks)

      # Sorts entries by (task_id, trial) for consistent ordering
      print(f"Sorting files in {folder_path}            --------------------------")
      cleaner.run_on_all_files_in_folder(cleaner.sort_by_task_id)
      
      print(f"Printing progress for {folder_path}    --------------------------")
      progress_viewer.progress_by_model()
      print()
      
    print(f"Printing detailed progress for {folder_path}    --------------------------")
    progress_viewer.detailed_progress(folder_path)
    print()

    if EVALUATE:
      print(f"Evaluating results in {folder_path}            --------------------------")
      evaluator.evaluate_folder()
      print()


# Test push