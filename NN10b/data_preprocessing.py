import pandas as pd
import torch
from scipy.signal import medfilt
from sklearn.model_selection import train_test_split


# normalize the data function
def load_processed_data():
    data = pd.read_excel("processed_music_data.xlsx")
    columns = list(data.drop(columns=['subject no.', 'label']).columns)
    col_dict = {i: col_name for i, col_name in enumerate(columns)}
    X = data.drop(columns=['subject no.', 'label']).values
    Y = data['label'].values
    # Split the data into training and testing sets (let's use 80% for training and 20% for testing)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    # to match the nn output
    Y_train_tensor -= 1
    Y_test_tensor -= 1
    return X_train_tensor, X_test_tensor,Y_train_tensor, Y_test_tensor, col_dict
def normalize_subjectwise(df):
    normalized_df = df.copy()
    for subject in df['subject no.'].unique():
        subject_data = df[df['subject no.'] == subject]
        for column in df.columns:
            if column not in ['subject no.', 'label']:
                min_val = subject_data[column].min()
                max_val = subject_data[column].max()
                normalized_df.loc[normalized_df['subject no.'] == subject, column] = (subject_data[column] - min_val) / (max_val - min_val)
    return normalized_df

# load data
eeg_features_df = pd.read_excel("../music-eeg-features.xlsx",skiprows=1)
# Normalize data for each subject

##### Data preparation
# Normalize
# X = eeg_features_df.drop(columns=['label']).values.astype(float)
# y = eeg_features_df['label'].values.astype(int).reshape(-1, 1)

normalized_data = normalize_subjectwise(eeg_features_df)
print(normalized_data.head())

# median filter, window = 3(default), and by diff by subject no
# Apply the median filter to each column except 'subject no.' and 'label', and ensure filtering is done per subject
for subject in normalized_data['subject no.'].unique():
    subject_data = normalized_data[normalized_data['subject no.'] == subject]
    for column in normalized_data.columns:
        if column not in ['subject no.', 'label']:
            normalized_data.loc[normalized_data['subject no.'] == subject, column] = medfilt(subject_data[column])

normalized_data.head()

# save
# Prefix each column with "normal_" except for 'subject no.' and 'label'
columns_to_rename = [col for col in normalized_data.columns if col not in ['subject no.', 'label']]
new_column_names = {col: "normal_" + col for col in columns_to_rename}
normalized_data = normalized_data.rename(columns=new_column_names)

# Save the processed data to an xlsx file
output_path = "./processed_music_data.xlsx"
normalized_data.to_excel(output_path, index=False)