package com.example.trafficsignrecognition

import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.*
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.trafficsignrecognition.ml.Model96
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private val pickImageRequestCode = 1000
    lateinit var labels: List<String>
    lateinit var model: Model96
    lateinit var imageProcessor: ImageProcessor

    @RequiresApi(Build.VERSION_CODES.M)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize labels and model
        labels = FileUtil.loadLabels(this, "labels.txt")
        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(32, 32, ResizeOp.ResizeMethod.BILINEAR)).build()
        model = Model96.newInstance(this)

        // Real-Time Detection button functionality
        findViewById<Button>(R.id.realTimeButton).setOnClickListener {
            startActivity(Intent(this, RealTimeActivity::class.java))
        }

        // Storage Detection button functionality
        findViewById<Button>(R.id.storageButton).setOnClickListener {
            openGallery()
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        startActivityForResult(intent, pickImageRequestCode)
    }

    @SuppressLint("SetTextI18n")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == pickImageRequestCode && resultCode == RESULT_OK && data != null) {
            val imageUri: Uri? = data.data
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, imageUri)
            val result = detectTrafficSign(bitmap)

            // Pass the image and the detection result to ResultActivity
            val resultIntent = Intent(this, ResultActivity::class.java)
            resultIntent.putExtra("DETECTED_IMAGE", result.first)  // Pass the processed image
            resultIntent.putExtra("DETECTION_RESULT", result.second)  // Pass the detection result text
            startActivity(resultIntent)
        }
    }
    private fun detectTrafficSign(bitmap: Bitmap): Pair<Bitmap, String> {
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(bitmap) }
        val processedImage = imageProcessor.process(tensorImage)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray
        val maxIndex = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.BLACK
            textSize = 20f
            style = Paint.Style.FILL
        }

        // Get screen height
        val screenHeight = Resources.getSystem().displayMetrics.heightPixels
        val bottomMargin = 50f
        val textHeight = 30f // Approximate text height
        var yPosition = screenHeight - bottomMargin // Positioning from the screen bottom

        var detectionResultText = ""

        if (maxIndex != -1 && outputFeature0[maxIndex] > 0.1) {
            val label = labels[maxIndex]
            val confidence = outputFeature0[maxIndex] * 100
            canvas.drawText("DETECTED: $label", 20f, yPosition, paint)
            yPosition += textHeight // Move down for the next text
            canvas.drawText("CONFIDENCE: ${"%.1f".format(confidence)}%", 20f, yPosition, paint)

            // Set the detection result text to be passed
            detectionResultText = "DETECTED: $label\nCONFIDENCE: ${"%.1f".format(confidence)}%"
        } else {
            canvas.drawText("DETECTED: UNKNOWN", 20f, yPosition, paint)
            detectionResultText = "DETECTED: UNKNOWN"
        }

        return Pair(mutableBitmap, detectionResultText)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) model.close()
    }
}