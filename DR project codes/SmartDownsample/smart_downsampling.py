# Importing required libraries
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd


# 1. Determine Target Dataset Size:
    # Define the desired size of your downsampled dataset (e.g., 1GB).
# 2. Analyze Original Dataset:
    # Perform a dry run query to estimate the size of the dataset with only relevant columns.
    # Calculate the number of rows needed to achieve the target size.
# 3. Preserve Class Balance:
    # Maintain the ratio of positive to negative cases.
    # Query the original dataset to get counts of positive and negative cases.
# 4. Create Downsampling Query:
    # Use UNION ALL to combine all positive cases with a random subset of negative cases.
    # Implement stratified sampling by using a WHERE clause with a random number generator (e.g., RND >= X).
# 5. Apply Weighting:
    # Add a weight column to compensate for the downsampling of negative cases.
    # Set the weight for positive cases to 1 and negative cases to the inverse of the sampling ratio.
# 6. Execute Downsampling:
    # Run the query to create a new table with the downsampled data.
# 7. Verify Results:
    # Check the size and row count of the new table.
    # Validate that the class balance is maintained.

# Set up BigQuery client
credentials = service_account.Credentials.from_service_account_file('./service_account.json')
client = bigquery.Client(credentials=credentials, project='Data Science')

# Define parameters
TARGET_SIZE_GB = 1
BASE_TABLE = f'your_project.your_dataset.your_table' # gannett-datascience.DR_Vitalii.wk_stg_6_sysdate
FEATURE_LIST = ['column1', 'column2', 'column3', 'target_column']
HOLDOUT_PCT = 0.1

# Step 1: Analyze original dataset
dry_query = f"""
SELECT {', '.join(FEATURE_LIST)}
FROM `{BASE_TABLE}`
"""

dry_job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
dry_job = client.query(dry_query, job_config=dry_job_config)
bytes_per_row = dry_job.total_bytes_processed / dry_job.total_rows

target_rows = int((TARGET_SIZE_GB * 1e9) / bytes_per_row)

# Step 2: Get class balance
balance_query = f"""
SELECT 
    SUM(CASE WHEN target_column > 0 THEN 1 ELSE 0 END) as positive_count,
    COUNT(*) as total_count
FROM `{BASE_TABLE}`
WHERE RAND() >= {HOLDOUT_PCT}
"""

balance_job = client.query(balance_query)
balance_results = balance_job.result().to_dataframe().iloc[0]
positive_count = balance_results['positive_count']
total_count = balance_results['total_count']
negative_count = total_count - positive_count

# Calculate sampling ratio for negative class
negative_sample_ratio = (target_rows - positive_count) / negative_count
negative_sample_threshold = 1 - negative_sample_ratio

# Step 3: Create and execute downsampling query
downsample_query = f"""
CREATE OR REPLACE TABLE `{BASE_TABLE}_downsampled` AS
SELECT {', '.join(FEATURE_LIST)}, 1 as weight
FROM `{BASE_TABLE}`
WHERE target_column > 0 AND RAND() >= {HOLDOUT_PCT}
UNION ALL
SELECT {', '.join(FEATURE_LIST)}, {1/negative_sample_ratio} as weight
FROM `{BASE_TABLE}`
WHERE target_column = 0 AND RAND() >= {max(HOLDOUT_PCT, negative_sample_threshold)}
"""

downsample_job = client.query(downsample_query)
downsample_job.result()  # Wait for the query to complete

# Step 4: Verify results
downsampled_table = client.get_table(f"{BASE_TABLE}_downsampled")
print(f"Downsampled table size: {downsampled_table.num_bytes / 1e9:.2f} GB")
print(f"Downsampled table rows: {downsampled_table.num_rows}")

# Step 5: Validate class balance
validate_query = f"""
SELECT 
    SUM(CASE WHEN target_column > 0 THEN weight ELSE 0 END) as weighted_positive_count,
    SUM(weight) as weighted_total_count
FROM `{BASE_TABLE}_downsampled`
"""

validate_job = client.query(validate_query)
validate_results = validate_job.result().to_dataframe().iloc[0]
print(f"Original positive ratio: {positive_count / total_count:.4f}")
print(f"Downsampled positive ratio: {validate_results['weighted_positive_count'] / validate_results['weighted_total_count']:.4f}")

