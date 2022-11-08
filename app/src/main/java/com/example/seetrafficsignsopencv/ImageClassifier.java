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

    public ImageClassifier(Context context) {
        this.imageSize = 320;
        this.context = context;

    }

    public ImageClass classifyImage(Mat image) {


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


            float[] confidences = outputFeature0.getFloatArray();


            for (float c : confidences){
                System.out.println("c0: " + c);
            }


//            for (float c : outputFeature1.getFloatArray()){
//                System.out.println("c1: " + c);
//            }
//            for (float c : outputFeature2.getFloatArray()){
//                System.out.println("c2: " + c);
//            }
//            for (float c : outputFeature2.getFloatArray()){
//                System.out.println("c3: " + c);
//            }



//            System.out.println("confidences 1: " + outputFeature1.getFloatArray());
//            System.out.println("confidences 2: " + outputFeature2.getFloatArray());
//            System.out.println("confidences 3: " + outputFeature3.getFloatArray());

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }









        // TÄSSÄ PITÄIS KEKSIÄ MIHIN LUOKKAAN KUVA KUULUU

        int randomClass = (int)(Math.random() * ImageClass.values().length);
        ImageClass imageClass = ImageClass.values()[randomClass];

        return imageClass;
    }


}

