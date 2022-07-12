# SWADE
Segmented WAvelet-DEnoising and stepwise linear fitting (SWADE) is one landslide dating technique that employs the principle that vegetation is often removed by landsliding in vegetated areas, causing a temporal decrease in normalized difference vegetation index (NDVI). The methods and results are illustrated in our publications in (Fu et al., 2022).

#### H4

## Flowchart of the methodology of the SWADE method

![figure2_SWAS_Code flow-print](https://user-images.githubusercontent.com/109142828/178498246-a5cff51e-66d7-4224-8e56-b98e38514d91.png)


One main file contains the SWADE code in the file of 'SWADE.py'. Three function files are 'b_read_Act_Axel_Lr.py', 'c_deno_wave.py' and'd_step_wise.py'. When run the main file, all the four files should be in the same file folder.

For people who knows how to download by using gee and geemap, one code to download our training data of NDVI timeseries is suggested in the file of 'DownloadNDVIFromGEETranslatedFromAxelCode.ipynb'.

To use this code more easily, one screenshot of filefolder is available . After users download and extract this file folder. Users can run the main code of 'SWADE', and the output figure will be saved in the file of 'output'.



