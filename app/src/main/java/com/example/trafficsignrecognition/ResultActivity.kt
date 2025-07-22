package com.example.trafficsignrecognition

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class ResultActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        // Get the detection result text passed from the previous activity
        val detectionResult = intent.getStringExtra("DETECTION_RESULT")

        // Get the processed image passed from the previous activity
        val detectedImage = intent.getParcelableExtra<Bitmap>("DETECTED_IMAGE")

        // Find the TextView in activity_result.xml and set the text
        val resultTextView = findViewById<TextView>(R.id.resultTextView)
        resultTextView.text = detectionResult

        // Find the ImageView to display the processed image
        val imageView = findViewById<ImageView>(R.id.resultImageView)
        imageView.setImageBitmap(detectedImage)
    }
}
