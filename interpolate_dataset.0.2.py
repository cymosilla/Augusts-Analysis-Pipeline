import data_cleaning
import glob
import pandas as pd

data_folder = '/Volumes/Samsung USB/data' #20T hard drive
code_folder = '~/Desktop/Augusts-Analysis-Pipeline' #this is only used to store a file with the number and name of any empty csv files for notekeeping etc. 
file_extension = 'newtracks02.csv'
max_seconds_gap = 2
actual_frames_per_second = 4.9

files = glob.glob(data_folder + f"/**/**/*{file_extension}", recursive=True)

#Run through all files of the above type and interpolate. If the file is empty, skip it.
empty_files = []

for file in files:
    print(file)
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        empty_files.append(file)
        print(f"File empty. number of empty files: {len(empty_files)}")
        continue

    if df.empty:
        empty_files.append(file)
        print(f"File empty. number of empty files: {len(empty_files)}")
        continue

    else:
        interpolated_df = data_cleaning.interpolate(df, max_seconds_gap, actual_frames_per_second)
        interpolated_df = data_cleaning.remove_jumps(interpolated_df)
        new_filename = file.replace(f'{file_extension}', '_interpolated.csv')
        interpolated_df.to_csv(new_filename, index=False)

print(f"Number of empty files: {len(empty_files)}")
empty_files_df = pd.DataFrame(empty_files)
empty_files_df.to_csv(code_folder + '/empty_files.csv', index=False)