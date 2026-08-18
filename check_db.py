import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect('state.db', check_same_thread=False)
memory = SqliteSaver(conn)
# list threads
cursor = conn.cursor()
cursor.execute("SELECT thread_id FROM checkpoints ORDER BY rowid DESC LIMIT 1;")
row = cursor.fetchone()
if row:
    thread_id = row[0]
    config = {"configurable": {"thread_id": thread_id}}
    state = memory.get(config)
    
    if state and hasattr(state, 'values'):
        vals = state.values
        print(f"--- THREAD ID: {thread_id} ---")
        print(f"User Request: {vals.get('user_request', '')}")
        print(f"Validation Valid?: {vals.get('is_valid')}")
        print(f"Retry Count: {vals.get('retry_count')}")
        print(f"Trust Label: {vals.get('trust_label')}")
        print(f"Integrity Passed?: {vals.get('resource_integrity_passed')}")
    else:
        print("Could not read state for thread:", thread_id)
else:
    print("No checkpoints found.")
