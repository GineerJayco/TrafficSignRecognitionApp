package com.example.trafficsignrecognition

import android.Manifest
import android.app.Activity
import android.content.ContentResolver
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.trafficsignrecognition.ml.Model96
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class StorageActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var model: Model96
    lateinit var labels: List<String>
    lateinit var imageProcessor: ImageProcessor

    // Register permission for reading external storage
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    // Register activity result to get image from gallery
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let { processImage(uri) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_storage)

        imageView = findViewById(R.id.selectedImage)

        val selectImageButton: Button = findViewById(R.id.selectImageButton)

        // Load labels and model
        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor = ImageProcessor.Builder().add(ResizeOp(32, 32, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = Model96.newInstance(this)

        // Check permissions (for Android 11 and above, we need to request runtime permission)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            selectImageButton.setOnClickListener {
                pickImageLauncher.launch("image/*")
            }
        } else {
            requestPermissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE)
        }
    }

    // Handle the image selection
    private fun processImage(uri: Uri) {
        try {
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            imageView.setImageBitmap(bitmap)
            detectImage(bitmap)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    // Function to run inference on the selected image
    private fun detectImage(bitmap: Bitmap) {
        // Resize the image to match the input size expected by the model
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true)

        // Normalize the image pixel values to be between 0 and 1
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)

        val imageProcessor = ImageProcessor.Builder().add(ResizeOp(32, 32, ResizeOp.ResizeMethod.BILINEAR)).build()
        val processedImage = imageProcessor.process(tensorImage)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        // Run inference
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        // Find the index of the max value in the output
        val maxIndex = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1

        // Confidence threshold to improve accuracy
        val confidenceThreshold = 0.8f  // Adjust this threshold as needed
        if (maxIndex != -1 && outputFeature0[maxIndex] > confidenceThreshold) {
            val label = labels[maxIndex]
            val probability = outputFeature0[maxIndex] * 100
            Toast.makeText(this, "Detected: $label with probability $probability%", Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(this, "No confident detection", Toast.LENGTH_SHORT).show()
        }
    }
}