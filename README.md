# LicensePlateReader

It is a project for the subject of Vision Systems conducted at Robots and Autonomous Systems specialization at the Poznan University of Technology.

## Author
Bartosz Ptak
    
## Task
Writing a program without using machine learning:
* find car plates on the image
* read the chars on them
    
## Assumptions
* law in Poland
* boards with only 7 characters
* images distortions for max. 45 degrees
   
## Final results on private test set:
* Find bbox accurancy: 91.67% (44 readed plates for 48 total)
* OCR accurancy: 97.78% (308 readed chars for 315 total)
* Total accurancy: 89.80% (308 good chars per 343 total)
* Execution time: 11.59s (per 48 images)
