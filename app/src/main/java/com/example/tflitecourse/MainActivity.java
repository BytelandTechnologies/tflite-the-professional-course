package com.example.tflitecourse;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;

/**
 * Main activity of the dog classification app developed during the Week 1 of the course
 * <em>TensorFlow Lite - The professional course</em>.
 * <p>
 *
 * @author Michel Meneses, Luiz Reis
 * @version 2.0
 * @since 1.0
 */

public class MainActivity extends AppCompatActivity {

    private static final int IMAGE_SIZE = 224;
    private static final int NUM_CLASSES = 120;
    private static final float CLASSIFICATION_THRESHOLD = 0.5f;
    private static final String MODEL_FILENAME = "model.tflite";
    private static final String LABELS_FILENAME = "labels.txt";

    private static final int REQUEST_CODE_IMAGE_CAMERA = 1;
    private static final int REQUEST_CODE_IMAGE_GALLERY = 100;
    private static final String LOG_TAG = MainActivity.class.getCanonicalName();

    private ImageView imageView;
    private Button buttonLoadImageFromGallery, buttonLoadImageFromCamera, buttonClassifyImage;
    private TextView textViewBreed, textViewConfidence;
    private Interpreter interpreter;
    private List<String> labels;
    private ImageProcessor imageProcessor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image);
        buttonLoadImageFromGallery = findViewById(R.id.button_load_image_from_gallery);
        buttonLoadImageFromCamera = findViewById(R.id.button_load_image_from_camera);
        buttonClassifyImage = findViewById(R.id.button_classify_image);
        textViewBreed = (TextView) findViewById(R.id.textView_dog);
        textViewConfidence = (TextView) findViewById(R.id.textView_confidence);

        try {
            this.initModel();
        } catch (IOException e) {
            Toast.makeText(this, "Initialization error!", Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }

        buttonLoadImageFromGallery.setOnClickListener(view -> openGalleryForImage());
        buttonLoadImageFromCamera.setOnClickListener(view -> openCameraForImage());
        buttonClassifyImage.setOnClickListener(view -> classifyImage());
    }

    /**
     * Set up the objects required to pre-process and classify the input images.
     */

    void initModel() throws IOException {
        this.labels = FileUtil.loadLabels(this, LABELS_FILENAME);
        this.interpreter = this.initInterpreter();
        this.imageProcessor = this.initImageProcessor();
    }

    /**
     * Loads the TF Lite model and encapsulate it inside a TF Lite interpreter.
     * <p>
     *
     * @return The TF Lite interpreter with the converted model already loaded.
     */

    private Interpreter initInterpreter() throws IOException {
        MappedByteBuffer tfLiteModel = FileUtil.loadMappedFile(this, MODEL_FILENAME);
        return new Interpreter(tfLiteModel, new Interpreter.Options());
    }

    /**
     * Set up a helper object used to pre-process the images to be classified.
     * <p>
     *
     * @return A helper object for pre-processing the input images.
     */

    private ImageProcessor initImageProcessor() {
        return new ImageProcessor.Builder()
                .add(new ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();
    }

    /**
     * Triggers an Android intent for launching the Gallery app.
     */

    private void openGalleryForImage() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_CODE_IMAGE_GALLERY);
    }

    /**
     * Triggers an Android intent for launching the Camera app.
     */

    private void openCameraForImage() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, REQUEST_CODE_IMAGE_CAMERA);
    }

    /**
     * Collects and displays the image returned either by the Gallery or the Camera app.
     */

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == REQUEST_CODE_IMAGE_GALLERY) {
                imageView.setImageURI(data.getData());
            } else if (requestCode == REQUEST_CODE_IMAGE_CAMERA) {
                Bundle extras = data.getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                imageView.setImageBitmap(imageBitmap);
            }
            buttonClassifyImage.setEnabled(true);
        }
    }

    /**
     * Runs the classification pipeline on the input image loaded via the Gallery or Camera app.
     * <p>
     * The classification pipeline includes pre-processing the input image, converting it into a
     * tensor, feeding the TF Lite interpreting with it and collecting the output prediction via
     * an output tensor buffer. Later, the final result is presented to the user by updating the
     * UI components accordingly.
     */

    private void classifyImage() {
        new Thread(() -> {
            Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
            if (bitmap == null) {
                Log.e(LOG_TAG, "Null bitmap!");
                return;
            }
            TensorImage inputImage = new TensorImage(DataType.FLOAT32);
            inputImage.load(bitmap);
            inputImage = imageProcessor.process(inputImage);
            TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);

            interpreter.run(inputImage.getBuffer(), output.getBuffer());

            TensorProcessor tensorProcessor = new TensorProcessor.Builder().build();
            TensorLabel tensorLabels = new TensorLabel(labels, tensorProcessor.process(output));
            Map<String, Float> floatMap = tensorLabels.getMapWithFloatValue();
            Map.Entry<String, Float> argMax = floatMap.entrySet().stream()
                    .max(Map.Entry.comparingByValue()).get();
            float confidence = argMax.getValue().floatValue();
            if (confidence > CLASSIFICATION_THRESHOLD)
                showOutput(argMax.getKey(), confidence);
            else
                showUnknownOutput();
        }).start();
    }

    /**
     * Updates the UI components with the latest classification results.
     */

    private void showOutput(String dogBreed, float confidence) {
        runOnUiThread(() -> {
            textViewBreed.setText(dogBreed);
            textViewConfidence.setText(String.format("%.2f%%", confidence));
        });
    }

    /**
     * Updates the UI components to inform the latest classification result is unknown.
     */

    private void showUnknownOutput() {
        runOnUiThread(() -> {
            textViewBreed.setText("-");
            textViewConfidence.setText(String.format("-"));
            Toast.makeText(this, "Unknown breed.", Toast.LENGTH_LONG).show();
        });
    }
}