package com.example.trafficsignrecognition

import android.annotation.SuppressLint
import android.graphics.*
import android.hardware.camera2.*
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import com.example.trafficsignrecognition.ml.Model96
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

@RequiresApi(Build.VERSION_CODES.LOLLIPOP)
class RealTimeActivity : AppCompatActivity() {

    lateinit var labels: List<String>
    lateinit var model: Model96
    lateinit var imageProcessor: ImageProcessor
    lateinit var textureView: TextureView
    lateinit var imageView: ImageView
    lateinit var cameraDevice: CameraDevice
    lateinit var cameraManager: CameraManager
    lateinit var handler: Handler

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_realtime)

        // Load model and labels
        labels = FileUtil.loadLabels(this, "labels.txt")
        model = Model96.newInstance(this)
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(32, 32, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        textureView = findViewById(R.id.textureView)
        imageView = findViewById(R.id.imageView)

        // Start camera setup
        setupCamera()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::model.isInitialized) {
            model.close()
        }
    }

    private fun setupCamera() {
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager

        val handlerThread = HandlerThread("CameraBackground").apply { start() }
        handler = Handler(handlerThread.looper)

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return true
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                processFrame(textureView.bitmap!!)
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {
        val cameraId = cameraManager.cameraIdList[0]
        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                startPreview()
            }

            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
            }

            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
            }
        }, handler)
    }

    private fun startPreview() {
        val surfaceTexture = textureView.surfaceTexture!!
        surfaceTexture.setDefaultBufferSize(textureView.width, textureView.height)
        val surface = Surface(surfaceTexture)

        val captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
        captureRequestBuilder.addTarget(surface)

        cameraDevice.createCaptureSession(
            listOf(surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    session.setRepeatingRequest(captureRequestBuilder.build(), null, handler)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {}
            },
            handler
        )
    }

    private fun processFrame(bitmap: Bitmap) {
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(bitmap) }
        val processedImage = imageProcessor.process(tensorImage)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 32, 32, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(processedImage.buffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        val maxIndex = outputFeature0.indices.maxByOrNull { outputFeature0[it] } ?: -1
        val maxProbability = if (maxIndex != -1) outputFeature0[maxIndex] else 0f

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            textSize = 40f
            style = Paint.Style.FILL
        }

        if (maxProbability > 0.9) {
            val label = labels[maxIndex]
            canvas.drawText("DETECTED: $label", 20f, 50f, paint)
            canvas.drawText("CONFIDENCE: ${"%.1f".format(maxProbability * 100)}%", 20f, 100f, paint)
        } else {
            canvas.drawText("DETECTED: UNKNOWN", 20f, 50f, paint)
        }

        imageView.setImageBitmap(mutableBitmap)
    }
}
