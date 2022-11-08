package com.example.seetrafficsignsopencv;

import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {

    private ImageClassifier imageClassifier;

    private static String LOGTAG = "OpenCV_Log";
    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.v(LOGTAG,"OpenCV loaded");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }

        }
    };

    Button btn_setting,btn_camera,btn_exit;
    FrameLayout frameLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageClassifier = new ImageClassifier(this);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);

        btn_setting = findViewById(R.id.btn_setting);
        btn_camera = (Button) findViewById(R.id.btn_camera);

        btn_exit=findViewById(R.id.btn_exit);


        btn_setting.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view){


                Toast.makeText(MainActivity.this, "You clicked settings.", Toast.LENGTH_SHORT).show();
            }
        });


        btn_camera.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                if(mOpenCvCameraView.getVisibility() == View.VISIBLE){
                    mOpenCvCameraView.setVisibility(View.INVISIBLE);


                }
                else{
                    mOpenCvCameraView.setVisibility(View.VISIBLE);
                }


                Toast.makeText(MainActivity.this, "You clicked camera.",Toast.LENGTH_SHORT).show();

            }

        });


        btn_exit.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view){
                moveTaskToBack(true);
                android.os.Process.killProcess(android.os.Process.myPid());
                System.exit(1);

            }

        });

    }

    @Override
    protected List<?extends CameraBridgeViewBase> getCameraViewList(){
        return Collections.singletonList(mOpenCvCameraView);
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {

        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            Mat input_rgba = inputFrame.rgba();

            ImageClass frameClass = imageClassifier.classifyImage(input_rgba);

            ImageView speedImage = (ImageView) findViewById(R.id.SLDisplay);
            // Päivittää UI:ta crashaa ilman
            runOnUiThread(new Runnable() {

                @Override
                public void run() {

                    // Stuff that updates the UI
                    // Updates speed limit image

                    if (frameClass == ImageClass.EMPTY) {
                        // Framesta ei löytyny merkkiä -> Ei vissiin tehä mitään
                        speedImage.setImageResource(ImageClass.EMPTY.id());

                    }
                    else {
                        // Tällä laitetaan luokittelun mukainen kuva näkyviin ruutuun
                        speedImage.setImageResource(frameClass.id());
                }
                }
            });

            /* MIKÄ TÄÄ ON JA TARVIIKO TÄTÄ? */
            Mat input_gray = inputFrame.gray();

            MatOfPoint corners = new MatOfPoint();
            Imgproc.goodFeaturesToTrack(input_gray,corners,20,0.01,10,new Mat(),3,false);
            Point[] cornercsArr = corners.toArray();

            for(int i = 0; i < corners.rows(); i++){
                Imgproc.circle(input_rgba,cornercsArr[i],10,new Scalar(0,255,0),2);
            }
            /* //MIKÄ TÄÄ ON? */


            return inputFrame.rgba();
        }
    };


    @Override
    public void onPause(){
        super.onPause();
        if(mOpenCvCameraView != null){
            mOpenCvCameraView.disableView();
        }
    }
    @Override
    public void onResume(){
        super.onResume();
        if(!OpenCVLoader.initDebug()){
            Log.d(LOGTAG,"OpenCV not found");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION,this,mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView != null){
            mOpenCvCameraView.disableView();
        }
    }


}