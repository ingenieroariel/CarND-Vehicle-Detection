##Vehicle Detection
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

This is it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extracted HOG features on cell #3 using the `hog` function from Scikit-Learn.

Here are two examples of `vehicle` and `non-vehicle`:

![alt text][image1]

I settled on the `YCrCb` colorspace after testing `RGB`, `HSV` and `YUV`. 

####2. Explain how you settled on your final choice of HOG parameters.

I did not do a lot of parameter tweaking on the HOG apart from what we got from the course. My experiments changing the pixes per cell to higher numbers to make the detection faster ended up backfiring with lower recall, so I stayed with (8, 8) pixes per cell and (32 x 32) spatial size. One important change was changing the histogram bins number from 32 in the examples to a lower (16) to lower the dimentionality and improve the classifier.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used a RobustScaler before feeding the data to the SVM  in cell `#4`, this took care of having zero mean and unitary standard deviation. I split data in 80% training and 20% testing and tried a few values of K. All in all, the testing most of the times reported very high numbers (above 98%) in testing but the training data was not well fitted for the example video, actual performance on the testing video is a lot lower than in testing.

Here are the actual parameters:

```
LinearSVC(C=100000, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to settle on 3 window sizes and aggregate the output in order to have a better heatmap. This resulted in better performance and results than just increasing the overlap value.

```
 for window_size in [(60, 60), (75, 75), (90, 90), (120, 120)]:
        
        raw_windows = slide_window(img, ...) 
    
        some_hot_windows = search_windows(raw_windows, ...)
        
        hot_windows = hot_windows + some_hot_windows

```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I created a video based pipeline demonstrator, it captures what each part of the pipeline sees and helped me a lot during debugging. The key insight as described before was using different window sizes, this helped group windows together on the car without creating a lot of false positives. Another thing that helped a lot was restricting the image to just the `y` pixels between `350` and `650` instead of the whole image. Similarly removing the left part of the image improved time and results but I ended up not doing it because that would simplify the problem too much and I should not assume that there are no cars on the curb.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a heatmap with all the positive detections and used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Since it took a few seconds per frame and a couple of hours to process the whole video with my pipeline, I also wrote to a file called labels.txt the bboxes per frame. I optimized to have total recall and make sure the cars were always detected on labels.txt. Futures iterations to improve precision and remove false positives can be done using that file. 

The heatmap and all the other steps can be seen in my  [video result](./output_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Main issues to work on in the future are the removal of false positives, and the inclusion of negative examples from the left side of the image. Anything detected there can be automatically fed to the SVM classifier to tweak it and make sure they are ignored. Improving the overlap on the sliding windows, in particular the Y axis is the best way to make the heatmap value jump up to 20 or 30 and successfully differentiate it from the noise in the left, unfortunately due to how long my pipeline takes to run I could not experiment more. It is important to note that a big part of the reason my pipeline takes a few seconds per image is due to the lot of `img.copy` and rendering done for the diagnosis. I tested YOLO v2 using darkflow and got way better and faster results, realtime on my GTX1070.
