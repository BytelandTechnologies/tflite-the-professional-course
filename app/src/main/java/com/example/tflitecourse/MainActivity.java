package com.example.tflitecourse;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
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
import java.util.Comparator;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE = 100;
    private static final int IMAGE_SIZE = 224;
    private static final String MODEL_FILENAME = "model.tflite";
    private static final String LABELS_FILENAME = "labels.txt";
    private static final String LOG_TAG = MainActivity.class.getCanonicalName();

    private ImageView imageView;
    private Button buttonLoadImage, buttonClassifyImage;
    private TextView textViewBreed, textViewConfidence;
    private Interpreter interpreter;
    private List<String> labels;
    private ImageProcessor imageProcessor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image);
        buttonLoadImage = findViewById(R.id.button_load_image);
        buttonClassifyImage = findViewById(R.id.button_classify_image);
        textViewBreed = (TextView) findViewById(R.id.textView_breed);
        textViewConfidence = (TextView) findViewById(R.id.textView_confidence);

        try {
            this.initModel();
        } catch (IOException e) {
            Toast.makeText(this, "Initialization error!", Toast.LENGTH_LONG).show();
            e.printStackTrace();
        }

        buttonLoadImage.setOnClickListener(view -> openGalleryForImage());
        buttonClassifyImage.setOnClickListener(view -> classifyImage());
    }

    void initModel() throws IOException {
        this.labels = FileUtil.loadLabels(this, LABELS_FILENAME);
        this.interpreter = this.initInterpreter();
        this.imageProcessor = this.initImageProcessor();
    }

    private Interpreter initInterpreter() throws IOException {
        MappedByteBuffer tfLiteModel = FileUtil.loadMappedFile(this, MODEL_FILENAME);
        return new Interpreter(tfLiteModel, new Interpreter.Options());
    }

    private ImageProcessor initImageProcessor() {
        return new ImageProcessor.Builder()
                .add(new ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(0, 255))
                .build();
    }

    private void openGalleryForImage() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK && requestCode == REQUEST_CODE) {
            imageView.setImageURI(data.getData());
            buttonClassifyImage.setEnabled(true);
        }
    }

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
            TensorBuffer output = TensorBuffer.createFixedSize(new int[]{1, 120}, DataType.FLOAT32);

            interpreter.run(inputImage.getBuffer(), output.getBuffer());

            TensorProcessor tensorProcessor = new TensorProcessor.Builder().build();
            TensorLabel tensorLabels = new TensorLabel(labels, tensorProcessor.process(output));
            Map<String, Float> floatMap = tensorLabels.getMapWithFloatValue();
            Map.Entry<String, Float> argMax = floatMap.entrySet().stream()
                    .max(Comparator.comparing(Map.Entry::getValue)).get();
            displayOutput(argMax.getKey(), argMax.getValue().floatValue());
        }).start();
    }

    private void displayOutput(String dogBreed, float confidence) {
        runOnUiThread(() -> {
            textViewBreed.setText(dogBreed);
            textViewConfidence.setText(String.format("%.2f%%", confidence));
        });
    }
}