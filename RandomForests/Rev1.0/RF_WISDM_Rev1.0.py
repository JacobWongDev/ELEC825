import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import glob

def load_arff_files(arff_folder_path):
    all_data = {
        'phone': {'accel': [], 'gyro': []},
        'watch': {'accel': [], 'gyro': []}
    }
    
    all_subjects = []
    
    devices = [d for d in os.listdir(arff_folder_path) if os.path.isdir(os.path.join(arff_folder_path, d))]
    print(f"Found devices: {devices}")
    
    for device in devices:
        device_path = os.path.join(arff_folder_path, device)
        sensors = [s for s in os.listdir(device_path) if os.path.isdir(os.path.join(device_path, s))]
        print(f"Found sensors for {device}: {sensors}")
        
        for sensor in sensors:
            sensor_path = os.path.join(device_path, sensor)
            arff_files = glob.glob(os.path.join(sensor_path, "*.arff"))
            print(f"Found {len(arff_files)} ARFF files for {device}/{sensor}")
            
            for file_path in arff_files:
                print(f"Processing: {os.path.basename(file_path)}")
                filename = os.path.basename(file_path)
                subject_id = filename.split('_')[1]
                if subject_id not in all_subjects:
                    all_subjects.append(subject_id)
                
                try:
                    data, meta = arff.loadarff(file_path)
                    df = pd.DataFrame(data)
                    
                    if len(all_data[device][sensor]) == 0:
                        print(f"Column names in first file: {df.columns.tolist()}")
                        print(f"Data types: {df.dtypes}")
                    
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].str.decode('utf-8')
                    
                    df['device'] = device
                    df['sensor'] = sensor
                    df['subject_id'] = subject_id
                    
                    all_data[device][sensor].append(df)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    for device in all_data:
        for sensor in all_data[device]:
            if all_data[device][sensor]:
                all_data[device][sensor] = pd.concat(all_data[device][sensor], ignore_index=True)
                print(f"Combined {device}/{sensor} data shape: {all_data[device][sensor].shape}")
            else:
                print(f"Warning: No data for {device}/{sensor}")
    
    return all_data, all_subjects

def prepare_data(all_data):
    device_sensor_dfs = []
    
    for device in all_data:
        for sensor in all_data[device]:
            if isinstance(all_data[device][sensor], pd.DataFrame) and not all_data[device][sensor].empty:
                df = all_data[device][sensor].copy()
                
                if 'ACTIVITY' in df.columns:
                    activity_col = 'ACTIVITY'
                elif 0 in df.columns or '0' in df.columns:
                    activity_col = 0 if 0 in df.columns else '0'
                    print(f"Using column {activity_col} as the activity column")
                else:
                    for col in df.columns:
                        if df[col].dtype == object or df[col].dtype == 'category':
                            unique_vals = df[col].unique()
                            print(f"Column {col} has unique values: {unique_vals[:10]}")
                            

                            if all(isinstance(val, str) and len(val) == 1 and 'A' <= val <= 'S' for val in unique_vals):
                                activity_col = col
                                print(f"Identified {col} as the activity column")
                                break
                    else:
                        for col in df.columns:
                            if 'activity' in str(col).lower():
                                activity_col = col
                                print(f"Using {col} as the activity column based on name")
                                break
                        else:
                            activity_col = df.columns[0]
                            print(f"Using first column {activity_col} as the activity column")
                
                y = df[activity_col].values
                print(f"Activity values sample: {y[:5]}")
                
                columns_to_drop = [activity_col, 'device', 'sensor', 'subject_id', 'class'] 
                df = df.drop(columns_to_drop, axis=1, errors='ignore')
                
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                print(f"Number of numeric feature columns: {len(numeric_cols)}")
                X = df[numeric_cols].values
                
                device_sensor_dfs.append((X, y, f"{device}_{sensor}"))
    
    all_X = []
    all_y = []
    
    for X, y, name in device_sensor_dfs:
        print(f"Shape of {name} data: {X.shape}")
        all_X.append(X)
        all_y.extend(y)
    
    if len(all_X) > 0:
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.array(all_y)
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_combined)
        
        print(f"Combined data shape: {X_combined.shape}")
        print(f"Combined labels shape: {y_encoded.shape}")
        print(f"Classes: {label_encoder.classes_}")
        print(f"Class distribution: {np.bincount(y_encoded)}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        return X_scaled, y_encoded, label_encoder.classes_
    else:
        raise ValueError("No valid data found")
    
import matplotlib.pyplot as plt

# def plot_activity_distribution(y, class_names):
#     # Count the occurrences of each activity label
#     unique_activities, counts = np.unique(y, return_counts=True)

#     # Plotting the activity distribution
#     plt.figure(figsize=(10, 6))
#     plt.bar(unique_activities, counts, color='skyblue')

#     # Setting x-ticks with class names
#     plt.xticks(unique_activities, class_names, rotation=45, ha='right')

#     # Adding labels and title
#     plt.xlabel('Activity')
#     plt.ylabel('Number of Instances')
#     plt.title('Activity Distribution')

#     # Show the plot
#     plt.tight_layout()
#     plt.show()

def main():
    BATCH_SIZE = 16
    arff_folder_path = r"C:\Users\rando\OneDrive\Desktop\Queens Coursework\ELEC 825 - MACHINE LEARNING AND DEEP LEARNING\Project\Datasets\WISDM_Dataset\wisdm-dataset\wisdm-dataset\arff_files"
    
    print("Loading ARFF files...")
    all_data, all_subjects = load_arff_files(arff_folder_path)
    
    print("Preparing data for training...")
    X, y, class_names = prepare_data(all_data)
    
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    
    # Train the model
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print(f"Random Forest Recall: {recall:.4f}")
    print(f"Random Forest F1 Score: {f1:.4f}")
    # After preparing the data (inside the main function), add this line:

    # plot_activity_distribution(y, class_names)


if __name__ == "__main__":
    main()
