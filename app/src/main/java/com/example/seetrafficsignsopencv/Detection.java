package com.example.seetrafficsignsopencv;

import org.opencv.core.Point;

public class Detection {
    ImageClass imgClass;
    Point startPoint;
    Point endPoint;

    public Detection(ImageClass imgClass, Point startPoint, Point endPoint) {
        this.imgClass = imgClass;
        this.startPoint = startPoint;
        this.endPoint = endPoint;
    }

}
