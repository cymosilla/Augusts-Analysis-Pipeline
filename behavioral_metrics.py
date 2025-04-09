import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import date
import os
from scipy import spatial
import glob

data_folder = "/Volumes/Samsung USB/data"
file_extension = "interpolated.csv" # the file extension for the data that you want to run behavioral metrics on
actual_frames_per_second = 4.9
speed_cutoff_seconds = 2
pixel_contact_distance = 206.1 # 206.1 pixels equals approximately 1cm in a BumbleBox
moving_threshold = 3.16

def compute_speed(df: pd.DataFrame, fps: int, speed_cutoff_seconds: int, moving_threshold: float, todays_folder_path: str, filename: str) -> pd.DataFrame:
 
    speed_cutoff_frames = fps*speed_cutoff_seconds
    # Sorting by ID and frame ensures that we compare positions of the same bee across consecutive frames
    df_sorted = df.sort_values(by=['ID', 'frame'])
   
    # Compute the difference between consecutive rows for centroidX and centroidY
    df_sorted['deltaX'] = df_sorted.groupby('ID')['centroidX'].diff()
    df_sorted['deltaY'] = df_sorted.groupby('ID')['centroidY'].diff()
    
    df_sorted['elapsed frames'] = df_sorted.groupby('ID')['frame'].diff()
    # Compute the Euclidean distance, which gives speed (assuming frame rate is constant)
    sub_df = df_sorted[ df_sorted['elapsed frames'] < speed_cutoff_frames ]
    sub_df['speed'] = np.sqrt(sub_df['deltaX']**2 + sub_df['deltaY']**2)

    #only calculate speed when moving, otherwise mark as NAN
    sub_df.loc[sub_df['speed'] < 3.16, 'speed'] = np.nan
    
    df_sorted.loc[:, 'speed'] = sub_df.loc[:, 'speed']
    # Drop temporary columns used for computations
    df_sorted.drop(columns=['deltaX', 'deltaY'], inplace=True)
    df_sorted.to_csv(todays_folder_path + "/" + filename + '_updated.csv', index=False)
    return df_sorted


def compute_activity(df: pd.DataFrame, fps: int, speed_cutoff_seconds: int, moving_threshold: float, todays_folder_path: str, filename: str) -> pd.DataFrame:

    speed_cutoff_frames = fps*speed_cutoff_seconds
    # Sorting by ID and frame ensures that we compare positions of the same bee across consecutive frames
    df_sorted = df.sort_values(by=['ID', 'frame'])

    # Compute the difference between consecutive rows for centroidX and centroidY
    df_sorted['deltaX'] = df_sorted.groupby('ID')['centroidX'].diff()
    df_sorted['deltaY'] = df_sorted.groupby('ID')['centroidY'].diff()

    df_sorted['elapsed frames'] = df_sorted.groupby('ID')['frame'].diff()
    # Compute the Euclidean distance, which gives speed (assuming frame rate is constant)
    sub_df = df_sorted[ df_sorted['elapsed frames'] < speed_cutoff_frames ]
    sub_df['activity'] = np.sqrt(sub_df['deltaX']**2 + sub_df['deltaY']**2) #Calculating speed here - we threshold for activity below

    sub_df.loc[sub_df['activity'] < moving_threshold, 'activity'] = 0 
    sub_df.loc[sub_df['activity'] <= moving_threshold, 'activity'] = 1

    df_sorted.loc[:, 'activity'] = sub_df.loc[:, 'activity']
    # Drop temporary columns used for computations
    df_sorted.drop(columns=['deltaX', 'deltaY'], inplace=True)
    df_sorted.to_csv(todays_folder_path + "/" + filename + '_updated.csv', index=False)
    return df_sorted


def compute_social_center_distance(df: pd.DataFrame, todays_folder_path: str, filename: str) -> pd.DataFrame:
    # Compute the social center for each frame
    #social_centers = df.groupby('frame')[['centroidX', 'centroidY']].mean()
    social_centers = df[['centroidX', 'centroidY']].mean() 
    print(social_centers[0])
    print(social_centers[1])
    #social_centers.columns = ['centerX', 'centerY']
    #print(social_centers['centerX'])
    #print(social_centers['centerY'])

    # Merge the social centers with the main dataframe to calculate distances
    #df = df.merge(social_centers, left_on='frame', right_index=True)

    # Compute the distance of each bee from the social center of its frame
    df['distance_from_center'] = np.sqrt((df['centroidX'] - social_centers[0])**2 + (df['centroidY'] - social_centers[1])**2)

    # Drop temporary columns used for computations
    #df.drop(columns=['centerX', 'centerY'], inplace=True)
    df_sorted = df.sort_values(by=['frame','ID'])
    df_sorted.to_csv(todays_folder_path + "/" + filename + '_updated.csv', index=False)
    return df_sorted



def pairwise_distance(df: pd.DataFrame, todays_folder_path: str, filename: str) -> pd.DataFrame:

    video_pd_df = 'None'
    for frame_num, frame_df in df.groupby(['frame']): #now for the subdataframe of each frame of a video of a colony, reset the index so its not a multiindex

        xy_array = frame_df[['centroidX','centroidY']].to_numpy() #make these two columns into an array of [x,y] for numpy to use
        dist_matrix = spatial.distance.pdist(xy_array)
        squareform = spatial.distance.squareform(dist_matrix) #turns the pairwise distance vector into a 2d array (which has double values but allows us to do row based operations)
        squareform[ squareform == 0 ] = np.nan
        pairwise_distance_df = pd.DataFrame(squareform, columns=frame_df['ID']) #make the squareform array into a dataframe, with the bee IDs as the columns
        bee_id_column = frame_df.loc[:,'ID'].reset_index(drop=True) #make a new bee ID column to add to the dataframe
        pairwise_distance_df.insert(0, 'ID', bee_id_column)
        pairwise_distance_df.columns.name = None #this stops bee ID showing up as the name of all the columns in the df, it looks weird and is confusing
        pairwise_distance_df['frame'] = int(frame_num[0])
        #pairwise_distance_df = pd.concat((frame_df['bee ID'], pd.DataFrame(squareform, columns=frame_df['bee ID'])), axis=1) #turns the squareform array into a dataframe indexed by bee ID
        #pairwise_distance_df.replace(0, np.nan, inplace=True) #replace zeros with nan values in order to exclude them from calculations 

        try:
            pairwise_distance_df['ID'] = pairwise_distance_df['ID'].astype('int')
        except ValueError:
            print('ValueError: are there any Nans in the bee IDs? Why would there be though...')
            print(frame_num, frame_df)
            return pairwise_distance_df['ID']

        if 'None' in video_pd_df:
            #print('avg_pd_df is equal to None!')
            video_pd_df = pairwise_distance_df

        else:
            
            pairwise_distance_df = pairwise_distance_df.loc[~pairwise_distance_df.index.duplicated(),:] #need these two lines to add to video level dataframe for some reason
            pairwise_distance_df = pairwise_distance_df.loc[:,~pairwise_distance_df.columns.duplicated()]
            try:
                video_pd_df = pd.concat([video_pd_df,pairwise_distance_df], axis=0, sort=True) #ignore_index=True)
            except:
                print(f"heres the videos index: {video_pd_df.index}")
                return 
    #calculate frame averages, mins, and maxes
    pairwise_distance_df = frame_avg_min_max_distances_to_other_bees(pairwise_distance_df)

    #extract frame_number column to bring it to the front of the df
    frame_column = video_pd_df['frame']
    bee_id_column = video_pd_df['ID']
    #drop the columns
    video_pd_df.drop(labels=['frame', 'ID'], axis=1, inplace=True)

    #then reinsert frame number at the beginning and averages at the end of the dataframe
    video_pd_df.insert(0, 'ID', bee_id_column)
    video_pd_df.insert(0, 'frame', frame_column)
    video_pd_df.reset_index(drop=True, inplace=True)
    video_pd_df.to_csv(todays_folder_path + '/' + filename + '_pairwise_distance.csv')

    return video_pd_df

def frame_avg_min_max_distances_to_other_bees(pairwise_distance_df):

    pairwise_v1 = pairwise_distance_df.drop(columns={'ID','frame'}) #drop non-data columns from new dataframe in order to calculate the min, max of the data without interference
    pairwise_distance_df['avg_distance'] = pairwise_v1.mean(axis=1, numeric_only=True, skipna=True)
    pairwise_distance_df['min_distance'] = pairwise_v1.min(axis=1, numeric_only=True, skipna=True) #makes sure to get the min from each row, excluding nans 
    pairwise_distance_df['max_distance'] = pairwise_v1.max(axis=1, numeric_only=True, skipna=True)

    return pairwise_distance_df

def contact_matrix(df: pd.DataFrame, todays_folder_path: str, pixel_contact_distance, filename: str):

    if df is None:
        return 1
    df_copy = df.copy() # make a copy to store the bee ID and frame number columns, which will be changed to a 1 or 0 in the original dataframe
    df[ df <= pixel_contact_distance ] = 1
    df[ df > pixel_contact_distance ] = 0
    df['ID'] = df_copy['ID'] # add the correct column back in by overwriting 'bee ID'
    df['frame'] = df_copy['frame'] # add the correct column back in by overwriting 'frame number'

    df.to_csv(todays_folder_path + '/' + filename + '_contacts.csv')

    return df


def calculate_behavior_metrics(df, actual_frames_per_second, moving_threshold, todays_folder_path, filename): #accept a path to a file or a path to a dataframe

    print("starting calculations")
    #not finished!
    if type(df) == str: #path to csv file
        df = pd.read_csv(df, header=0)
        print("Reading csv")
        
    elif type(df) == pd.DataFrame: #dataframe 
        #df = data_object
        print("Its a dataframe!")
        

    print("Trying speed")
    #try:
    df = compute_speed(df,actual_frames_per_second,4, moving_threshold, todays_folder_path, filename)
    print('just computed speed')
    #except Exception as e:
    #    print("Exception occurred: %s", str(e))
    #    logger.debug("Exception occurred: %s", str(e))
                
    print("Trying activity")
    df = compute_activity(df,actual_frames_per_second,4, moving_threshold, todays_folder_path, filename)
    print("Just computed activity")
        
    try:
        df = compute_social_center_distance(df, todays_folder_path, filename)
        print('just computed distance from center')
    except Exception as e:
        print("Exception occurred: %s", str(e))
        #logger.debug("Exception occurred: %s", str(e))
            
            

    print("Trying pairwise distance")
    pw_df = pairwise_distance(df, todays_folder_path, filename)
        
            
            

    print("Trying contacts")
    contact_df = contact_matrix(pw_df, todays_folder_path, pixel_contact_distance, filename)



def main():

    for file in glob.glob(data_folder + f"/**/*{file_extension}", recursive=True):
        
        print(file)
        
        filename = os.path.basename(file)
        dirname = os.path.dirname(file)
        print(filename)
        print(dirname)
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print("Exception occurred: %s", str(e))
            continue
        
        filename = filename.replace(file_extension, "")
        ''' check whether files have been interpolated yet or not '''
        calculate_behavior_metrics(df, actual_frames_per_second, moving_threshold, dirname, filename)


if __name__ == '__main__':
    main()