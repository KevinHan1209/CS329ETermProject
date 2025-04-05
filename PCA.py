import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/kevinhan/CS329ETermProject/koi_data.csv'
data = pd.read_csv(file_path)

data = data.drop(columns=['koi_disposition'])

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame for the principal components
pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
pca_df = pd.DataFrame(data=principal_components, columns=pca_columns)

# Save the PCA results to a new CSV file
output_path = '/Users/kevinhan/CS329ETermProject/pca_results.csv'
pca_df.to_csv(output_path, index=False)

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)