import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

 
# From Python3.7 you can add
# keyword argument capture_output
print(subprocess.run(["/localhome/pagand/projects/e2etransfuser/leaderboard/scripts/run_evaluation.sh"],
                     capture_output=True))

