
import os
from progress import ProgressViewer
from clean import Cleaner


if __name__ == "__main__":
  
  model_size = "4B" # 4B, 8B, 14B, 32B
  env = "airline" # retail, airline
  strategy = "react" # act, react, fc
  folder_path = f"results/{env}/{strategy}/{model_size}"

  progress_viewer = ProgressViewer(model_size, env, strategy, folder_path)
  cleaner = Cleaner(model_size, env, strategy, folder_path)

  print(f"Sorting files in {folder_path}            --------------------------")
  cleaner.run_on_all_files_in_folder(cleaner.sort_by_task_id)
  print(f"Removing error logs from {folder_path}    --------------------------")
  cleaner.run_on_all_files_in_folder(cleaner.remove_error_logs)

  # for i in range(1, 6):
  #   cleaner.task_differences(f"num_trials-{i}.json")


  progress_viewer.progress_by_model()
  # progress_viewer.detailed_progress() # more fined grained view of progress