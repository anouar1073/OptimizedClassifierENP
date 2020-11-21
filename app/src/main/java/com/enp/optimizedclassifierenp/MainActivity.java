package com.enp.optimizedclassifierenp;


import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;



import android.database.Cursor;
import android.app.ActivityManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.media.Image;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static int ACTIVITY_START_CAMERA_APP = 1;
    private static int RESULT_LOAD_IMAGE= 0;

    float[] mean = {0.485f, 0.456f, 0.406f};
    float[] std = {0.229f, 0.224f, 0.225f};

    Bitmap bitmap = null;
    Module module = null;
    long time_elapsed=0;
    long time_elapsed_load=0;
    Tensor input;
    Runtime runtime = Runtime.getRuntime();
    long usedMemInMB_before;
    long usedMemInMB_after;
    boolean loaded= false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getSupportActionBar().setTitle("Classifier Demo");
        ImageButton loadButton = (ImageButton) findViewById(R.id.button);
        Button detectButton = (Button) findViewById(R.id.detect);
        ImageButton captureButton = (ImageButton) findViewById((R.id.capture));
        //final TextView profiling_stats= (TextView) findViewById(R.id.profiling_show);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
        if(!loaded) {
            try {
                //usedMemInMB_before=memoryInfo.availMem/1048576L;
                //usedMemInMB_before=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                long start = System.currentTimeMillis();
                module = Module.load(fetchModelFile(MainActivity.this, "resnet_baseline.pt"));
                time_elapsed_load = System.currentTimeMillis() - start;
                //usedMemInMB_after=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                //usedMemInMB_after = memoryInfo.availMem/1048576L;

            } catch (IOException e) {
                finish();
            }
        }

        captureButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent();
                i.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

                startActivityForResult(i, ACTIVITY_START_CAMERA_APP);
            }
        });

        loadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                TextView textView = findViewById(R.id.result_text);
                textView.setText("");
                Intent i = new Intent(
                        Intent.ACTION_PICK,
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

                startActivityForResult(i, RESULT_LOAD_IMAGE);
            }
        });

        detectButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View arg0) {
                // bitmap
                //Getting the image from the image view
                ImageView imageView = (ImageView) findViewById(R.id.image);

                try {

                    //Read the image as Bitmap
                    bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();

                    //Here we reshape the image into 400*400
                    bitmap = Bitmap.createScaledBitmap(bitmap, 400, 400, true);

                } catch (Exception e) {
                    finish();
                }


                input = TensorImageUtils.bitmapToFloat32Tensor(
                        bitmap,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB
                );


                //Loading the model file.
                //Input Tensor


                //Calling the forward of the model to run our input
                long start =System.currentTimeMillis();
                //final long usedMemInMB_before=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                final Tensor output = module.forward(IValue.from(input)).toTensor();
                //final long usedMemInMB_after=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                time_elapsed = System.currentTimeMillis() - start;


                final float[] score_arr = output.getDataAsFloatArray();

                // Fetch the index of the value with maximum score
                float max_score = -Float.MAX_VALUE;
                int ms_ix = -1;
                for (int i = 0; i < score_arr.length; i++) {
                    if (score_arr[i] > max_score) {
                        max_score = score_arr[i];
                        ms_ix = i;
                    }
                }

                //Fetching the name from the list based on the index
                String detected_class = ModelClasses.MODEL_CLASSES[ms_ix];

                //Writing the detected class in to the text view of the layout
                TextView textView = findViewById(R.id.result_text);
                textView.setText( "Class: " + detected_class);
                textView.setText(textView.getText() + "\n"
                        + " inference time: "+ Long.toString(time_elapsed)+ "ms" + "\n"
                        + "loading time: "+ Long.toString(time_elapsed_load)+"ms"
                );


            }
        });

    }
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflaster = getMenuInflater();
        inflaster.inflate(R.menu.app_menu, menu);
        return true;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        //This functions return the selected image from gallery
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == ACTIVITY_START_CAMERA_APP && resultCode == RESULT_OK && null != data) {
            Bundle extras = data.getExtras();
            Bitmap photoCapturedBitmap = (Bitmap) extras.get("data");

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(photoCapturedBitmap);
        }
        else if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            String[] filePathColumn = { MediaStore.Images.Media.DATA };

            Cursor cursor = getContentResolver().query(selectedImage,
                    filePathColumn, null, null, null);
            cursor.moveToFirst();

            int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
            String picturePath = cursor.getString(columnIndex);
            cursor.close();

            ImageView imageView = (ImageView) findViewById(R.id.image);
            imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));

            //Setting the URI so we can read the Bitmap from the image
            imageView.setImageURI(null);
            imageView.setImageURI(selectedImage);

        }
    }

    public static String fetchModelFile(Context context, String modelName) throws IOException {
        File file = new File(context.getFilesDir(), modelName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(modelName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.baseline:
                try {
                    //usedMemInMB_before=memoryInfo.availMem/1048576L;
                    //usedMemInMB_before=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                    long start =System.currentTimeMillis();
                    module = Module.load(fetchModelFile(MainActivity.this, "resnet_baseline.pt"));
                    time_elapsed_load = System.currentTimeMillis() - start;
                    //usedMemInMB_after=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                    //usedMemInMB_after = memoryInfo.availMem/1048576L;

                } catch (IOException e) {
                    finish();
                }
                loaded=true;
                break;
            case R.id.omtimized:
                try {
                    //usedMemInMB_before=memoryInfo.availMem/1048576L;
                    //usedMemInMB_before=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                    long start =System.currentTimeMillis();
                    module = Module.load(fetchModelFile(MainActivity.this, "resnet_q_.pt"));
                    time_elapsed_load = System.currentTimeMillis() - start;
                    //usedMemInMB_after=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
                    //usedMemInMB_after = memoryInfo.availMem/1048576L;

                } catch (IOException e) {
                    finish();
                }
                loaded=true;
                break;
        }
        return true;
    }
}

