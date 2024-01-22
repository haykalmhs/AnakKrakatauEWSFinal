import pandas as pd
import numpy as np
import librosa
import glob
import os
from tqdm import tqdm

PATH_DATA = "/path/to/csv/files"
OUTPUT_DIR = "/path/to/output/directory"
WINDOW_SIZE = 3000
WINDOW_STEP = 3000  # This will determine the overlap; set less than WINDOW_SIZE for overlap
N_FFT = 256
HOP_LENGTH = 128

def preprocess_data(folder_path=PATH_DATA, output_dir=OUTPUT_DIR, window_size=WINDOW_SIZE, window_step=WINDOW_STEP, n_fft=N_FFT, hop_length=HOP_LENGTH):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_metadata = []
    for csv_file in tqdm(glob.glob(folder_path + '/*.csv'), desc='Processing files'):
        df = pd.read_csv(csv_file, index_col='time', parse_dates=['time'])
        base_name = os.path.splitext(os.path.basename(csv_file))[0]

        for start in range(0, len(df) - window_size + 1, window_step):
            end = start + window_size
            window_data = df.iloc[start:end]
            window = window_data['beam [Pa]'].values
            Zxx = librosa.stft(window, n_fft=n_fft, hop_length=hop_length)
            Zxx_magnitude = np.abs(Zxx)[..., np.newaxis]
            output_file_path = os.path.join(output_dir, f"{base_name}_window_{len(all_metadata)}.npy")
            np.save(output_file_path, Zxx_magnitude)
            
            median_time = window_data.index[len(window_data)//2] if len(window_data) % 2 == 1 else \
                          window_data.index[(len(window_data)//2)-1] + (window_data.index[len(window_data)//2] - window_data.index[(len(window_data)//2)-1]) / 2

            all_metadata.append({
                'file_name': base_name,
                'window_index': len(all_metadata),
                'median_time': str(median_time),
                'file_path': output_file_path,
            })

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df['median_time'] = pd.to_datetime(metadata_df['median_time'])
    metadata_df.sort_values(by='median_time', inplace=True)
    metadata_df.to_csv(os.path.join(output_dir, 'combined_metadata.csv'), index=False)

    return metadata_df