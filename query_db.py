import sqlite3

def check_runs():
    db_path = 'C:/Users/enosh/oralguard/mlflow/mlflow.db'
    conn = sqlite3.connect(db_path)
    
    # 1. Print recent runs
    runs = conn.execute("SELECT run_uuid, name, status, start_time FROM runs ORDER BY start_time DESC LIMIT 3").fetchall()
    print("Recent Runs:")
    for r in runs:
        print(f"  ID: {r[0]} | Name: {r[1]} | Status: {r[2]} | StartTime: {r[3]}")
        
    # 2. Print latest metrics for the most recent run
    if runs:
        # 2. Print historical metrics for the most recent run
        import pandas as pd
        history = pd.read_sql_query(
            f"SELECT step, key, value FROM metrics WHERE run_uuid = '{runs[0][0]}' ORDER BY step, key", 
            conn
        )
        print(f"\nHistorical Metrics for {runs[0][1]}:")
        if not history.empty:
            # Pivot table to show step as rows and keys as columns
            pivoted = history.pivot(index='step', columns='key', values='value')
            print(pivoted.to_string())
        else:
            print("No metrics history recorded yet.")
            
    conn.close()

if __name__ == "__main__":
    check_runs()
