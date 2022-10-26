package com.example.seetrafficsignsopencv;

import org.opencv.core.Mat;

public class ImageClassifier {

    private Object tekoälymalli;
    // jne.... Tähän kaikki mitä tekoälyn käyttö vaatii

    public ImageClassifier() {


    }

    public ImageClass classifyImage(Mat image) {

        // TÄSSÄ PITÄIS KEKSIÄ MIHIN LUOKKAAN KUVA KUULUU

        int randomClass = (int)(Math.random() * ImageClass.values().length);
        ImageClass imageClass = ImageClass.values()[randomClass];

        return imageClass;
    }


}

