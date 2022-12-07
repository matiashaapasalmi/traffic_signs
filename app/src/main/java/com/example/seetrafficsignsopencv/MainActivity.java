package com.example.seetrafficsignsopencv;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.preference.Preference;
import androidx.preference.PreferenceManager;

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
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.v(LOGTAG, "OpenCV loaded");
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }

        }
    };

    Button btn_setting, btn_camera, btn_exit;
    FrameLayout frameLayout;
    Boolean debug_mode;
    String model;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        debug_mode = (prefs.getBoolean("debug_mode", true));
        model = (prefs.getString("models","CDC"));
        Log.d("MODEL",model);



        imageClassifier = new ImageClassifier(this);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);

        btn_setting = findViewById(R.id.btn_setting);
        btn_camera = (Button) findViewById(R.id.btn_camera);

        btn_exit = findViewById(R.id.btn_exit);

        btn_setting.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(MainActivity.this, SettingsActivity.class);
                startActivity(intent);
            }
        });

        btn_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (mOpenCvCameraView.getAlpha() == 0) {
                    mOpenCvCameraView.setAlpha(1);
                } else {
                    mOpenCvCameraView.setAlpha(0);
                }
           }
        });

        btn_exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                moveTaskToBack(true);
                android.os.Process.killProcess(android.os.Process.myPid());
                System.exit(1);
            }
        });

    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
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

            try {

                Mat input_rgba = inputFrame.rgba();

                Detection detection = null;

                System.out.println("Detection with: " + model);

                if (model.equals("640BW")) {
                    detection = imageClassifier.classifyImageBWLarge(input_rgba);
                } else if (model.equals("640C")) {
                    detection = imageClassifier.classifyImageColorLarge(input_rgba);
                } else if (model.equals("CDC")) {
                    detection = imageClassifier.classifyImageCDC(input_rgba);
                } else if (model.equals("320BW")) {
                    detection = imageClassifier.classifyImageBWSmall(input_rgba);
                } else if (model.equals("320C")) {
                    detection = imageClassifier.classifyImageColorSmall(input_rgba);
                } else {
                    detection = imageClassifier.classifyImage(input_rgba);
                } // 320C

                ImageClass frameClass = detection.imgClass;

                ImageView speedImage = (ImageView) findViewById(R.id.SLDisplay);
                // Päivittää UI:ta crashaa ilman
                runOnUiThread(new Runnable() {

                    @Override
                    public void run() {

                        // Stuff that updates the UI
                        // Updates speed limit image

                        if (frameClass != ImageClass.EMPTY) {
                            // Tällä laitetaan luokittelun mukainen kuva näkyviin ruutuun
                            speedImage.setImageResource(frameClass.id());
                        }
                    }
                });

                if (frameClass != ImageClass.EMPTY && debug_mode) {
                    Imgproc.rectangle(input_rgba, detection.startPoint, detection.endPoint, new Scalar(255, 222, 0), 3);

                    int confidence = (int) (detection.confidence * 100 + 0.5);

                    Imgproc.putText(input_rgba, frameClass.toString() + " " + confidence + " %", new Point(10, 50),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 1.5, new Scalar(255, 222, 0), 2, Imgproc.LINE_AA, false);
                }
                return input_rgba;
            }
            catch (Exception e){
                return inputFrame.rgba();
            }
        }
    };


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(LOGTAG, "OpenCV not found");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }


}