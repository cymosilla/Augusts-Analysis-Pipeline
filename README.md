At this point, you should have completed collecting your data from the BumbleBoxes

Assumptions for the following analysis:
You have recorded videos, but you haven't tracked them yet. This means that on the BumbleBoxes, you've
set:
tag_tracking = False
track_recorded_videos = False

Next assumption: You want to interpolate missing data under a certain number of seconds (usually 2 or 3)





Steps:
1. Open a terminal
2. Enter Augusts Analysis Pipeline folder:
cd ~/Desktop/Augusts-Analysis-Pipeline

2. Type the following into the terminal: 
source virtualbee/bin/activate

3. Find the path to the USB or external volume you are using - this can be done by right-clicking on a folder inside the volume, 
holding down the Cmd key, and clicking 'copy X as pathname'
Then you can paste the path to that folder somewhere, like a text document like this one, to get the following:

/Volumes/Samsung USB/data

With this info, you know that /Volumes/Samsung USB/ is the path to your volume. Note that any spaces in a path name usually
need to have quotes around them when we type them in, so it should look like this instead:

/Volumes/'Samsung USB'

4. In the same terminal as was used for the previous steps, type the following to run the tracking script:

python3 track-tags-to-csv.0.11.py --volume /Volumes/'Samsung USB' --folder data --extension newtracks02.csv (you decide what extension you want or if you want one)

## Output
This analyzes videos in the Samsung USB folder under different dates & 

5. That will track the tags and create csv files, next we need to interpolate that data. To do this, run the interpolate_dataset.0.2.py after setting the relevant variables at the top of the script. 

python3 interpolate_dataset.0.2.py

6. Next, run the behavioral metrics script on the interpolated data, after first setting the relevant variables at the top of the script. 

python3 behavioral_metrics.py

7. For each day, you might want to compute a composite nest image that removes bees. To do this, you should run generate_nest_image.py after setting all the relevant variables at the top of the script. You will need to run it for each day and then manually change the data_folder_path variable. The shuffle variable picks a random assortment of images from throughout the day, which is usually better, so I would recommend turning this to True. 



## Outputs



Tag tracking optimization script:

Hi Carm, I also got the tag tracking optimization script working, which means that you can figure out which Aruco parameters are the best for tracking the tags in the track-tags-to-csv script. You'll need to manually change them, but I used this video: 

bumblebox-03_2025-04-07_16_50_36.mp4 (it's on the Desktop)

This video has 47 frames in it where 5 tags should be visible, and 54 frames where only 4 tags are visible. Thus the best tracking score we should expect while minimizing duplicate tag tracks or erroneous tag readings is 4.46 tags per frame being read on average. The math is just (47*5) + (54*4) / 101 (total number of frames in the video)

So we're looking for aruco parameters that minimize the tracking time necessary and hit approximately 4.46 tag reads per frame. The script is currently running and hasn't finished, 
but these parameters are looking pretty good so far:

Params: (0.02, 3, 31, 3, 0.08) -> Avg markers: 4.46, Time: 7.54s 
That would mean the following:

Optimal ArUco Tracking Parameters:
minMarkerPerimeterRate: 0.02
adaptiveThreshWinSizeMin: 3
adaptiveThreshWinSizeMax: 31
adaptiveThreshWinSizeStep: 3
polygonalApproxAccuracyRate: 0.08
Average markers detected: 4.46
Total time: 7.54s

If these hold up and are the best parameters for the BumbleBox, you'll need to change the parameters in the tracking function, like I said above - let me know if you need any help!


TO DO:
Someone in the lab also need to turn off tag tracking on all the Pis in the setup.py script within the BumbleBox folder on the Desktop of each Pi -

Set:
tag_tracking = False
track_recorded_videos = False
interpolate_data = False
calculate_behavioral_metrics = False


