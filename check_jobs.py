from db.job_store import load_all_jobs
import pprint
jobs = load_all_jobs(limit=1)
if jobs:
    print("Latest Job ID:", jobs[0]['id'])
    print("Prompt:", jobs[0]['prompt'])
    print("Trust Label:", jobs[0]['trust_label'])
    print("Workflow:", jobs[0]['workflow'])
else:
    print("No jobs found.")
