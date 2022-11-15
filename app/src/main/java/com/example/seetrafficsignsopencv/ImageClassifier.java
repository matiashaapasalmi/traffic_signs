package com.example.seetrafficsignsopencv;

import android.content.Context;
import android.graphics.Bitmap;

import com.example.seetrafficsignsopencv.ml.Detect;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ImageClassifier {

    private final Context context;

    private int imageSize;

    private float detectionThreshold;

    public ImageClassifier(Context context) {
        this.imageSize = 320;
        this.detectionThreshold = (float) 0.20;
        this.context = context;

    }

    public ImageClass classifyImage(Mat image) {

        ImageClass imageClass = ImageClass.EMPTY;

        try {
            Detect model = Detect.newInstance(context);

            // Konvertoidaan Mat -> Bitmap

            Bitmap bitmap = Bitmap.createBitmap(image.cols(), image.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(image, bitmap);
            bitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false);

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, imageSize, imageSize, 3}, DataType.FLOAT32);

            // Työnnetään bitmap inputfeatureen
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int [] intValues = new int[imageSize * imageSize];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

            int pixel = 0;
            for(int i = 0; i < imageSize; i++){
                for(int j = 0; j < imageSize; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Detect.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
            TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
            TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();


            float[] confidences     = outputFeature0.getFloatArray();
            float[] detectionPoints = outputFeature1.getFloatArray();
            int[]   detections      = outputFeature3.getIntArray();

            int bestIndex = detections[0];
            float bestConfidence = confidences[0];

            imageClass = ImageClass.values()[bestIndex];

            System.out.println("DETECTION DEBUG: ImageClass: " + imageClass);
            System.out.println("DETECTION DEBUG: Confidence: " + bestConfidence);

            if(bestConfidence < this.detectionThreshold) {
                System.out.println("DETECTION DEBUG: Confidence under threshold. Setting output to EMPTY");
                imageClass = ImageClass.EMPTY;
            }

            model.close();

        } catch (IOException e) {
            // TODO Handle the exception
        }

        return imageClass;

    }


}

