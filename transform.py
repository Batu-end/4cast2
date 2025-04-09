import pandas as pd

# Load the CSV file
input_file = 'DataCleaned.csv'  # Replace with your actual file path
output_file = 'updated_dataset.csv'  # Replace with your desired output file path

# Read the CSV file
df = pd.read_csv(input_file)

# Replace the first column (assumed to be dates) with indexes
df.iloc[:, 0] = range(len(df))

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated dataset saved to {output_file}")



