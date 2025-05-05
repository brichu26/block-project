import csv

# File paths
official_repos_name_file = 'official_repos_name.csv'
pulsemcp_servers_all_file = 'pulsemcp_servers_all.csv'
output_file = 'filtered_servers.csv'

# Read official repo names into a set for fast look-up
with open(official_repos_name_file, mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    official_repos = {f"{row['owner']},{row['repo']}" for row in csvreader}

# Read pulsemcp_servers_all.csv and filter based on official repo names
with open(pulsemcp_servers_all_file, mode='r') as csvfile:
    csvreader = csv.DictReader(csvfile)
    header = csvreader.fieldnames
    filtered_rows = []

    for row in csvreader:
        server_key = f"{row['owner']},{row['repo']}"
        if server_key in official_repos:
            filtered_rows.append(row)

# Write the filtered data to a new CSV file
with open(output_file, mode='w', newline='') as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=header)
    csvwriter.writeheader()
    csvwriter.writerows(filtered_rows)

print(f"Filtered data has been written to {output_file}")