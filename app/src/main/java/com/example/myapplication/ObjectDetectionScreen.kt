package com.example.myapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager.PERMISSION_GRANTED
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.example.myapplication.Detection
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
//import org.tensorflow.lite.support.image.ops.NormalizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.BufferedReader
import java.io.InputStreamReader

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.material3.Button
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.core.content.ContextCompat
import androidx.core.app.ActivityCompat.startActivityForResult
import androidx.lifecycle.DefaultLifecycleObserver

@Composable
fun ObjectDetectionScreen() {
    val context = LocalContext.current
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var detectionResult by remember { mutableStateOf<List<Detection>>(emptyList()) }

    val lifecycleOwner = LocalLifecycleOwner.current
    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == androidx.core.content.PermissionChecker.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { granted ->
            hasCameraPermission = granted
        }
    )

    // Using DefaultLifecycleObserver
    DisposableEffect(lifecycleOwner) {
        val observer = object : DefaultLifecycleObserver {
            override fun onStart(owner: androidx.lifecycle.LifecycleOwner) {
                if (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.CAMERA
                    ) != androidx.core.content.PermissionChecker.PERMISSION_GRANTED
                ) {
                    launcher.launch(Manifest.permission.CAMERA)
                }
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    // Camera Launcher
    val captureLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicturePreview()) { result ->
        bitmap = result
    }

    // Object Detection
    LaunchedEffect(bitmap) {
        if (bitmap != null) {
            withContext(Dispatchers.Default) {
                detectionResult = detectObjects(context, bitmap!!)
            }
        }
    }

    Column(Modifier.fillMaxSize()) {
        if (bitmap != null) {
            Box(Modifier.weight(1f)) {
                Image(bitmap = bitmap!!.asImageBitmap(), contentDescription = null, modifier = Modifier.fillMaxSize())
                detectionResult.forEach { detection ->
                    Box(
                        Modifier
                            .offset(detection.x.dp, detection.y.dp)
                            .size(detection.width.dp, detection.height.dp)
                            .border(2.dp, Color.Red)
                            .zIndex(1f)
                    ) {
                        Text(detection.label, color = Color.White, modifier = Modifier.padding(4.dp))
                    }
                }
            }
        }

        // Capture Button
        Button(
            onClick = {
                if (hasCameraPermission) {
                    captureLauncher.launch(null)
                } else {
                    // Handle permission not granted (show a message, etc.)
                }
            },
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            enabled = hasCameraPermission
        ) {
            Text("Capture Image")
        }
    }
}

fun detectObjects(context: Context, image: Bitmap): List<Detection> {
    val model = Interpreter(FileUtil.loadMappedFile(context, "best_float32.tflite"))
    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(640, 640, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    var tensorImage = TensorImage(DataType.FLOAT32)
    tensorImage.load(image)
    tensorImage = imageProcessor.process(tensorImage)

    val inputArray = arrayOf(tensorImage.buffer)
    val outputMap = mutableMapOf<Int, Any>()
    val outputArray = Array(1) { Array(25) { FloatArray(8400) } } // Sesuaikan dengan bentuk output model
    outputMap[0] = outputArray
    model.runForMultipleInputsOutputs(inputArray, outputMap)

    val classes = readClasses(context)
    val detections = mutableListOf<Detection>()

    // Mengambil nilai numDetections
    val numDetections = outputArray[0][0][3].toInt()

    for (i in 0 until numDetections) {
        val confidence = outputArray[0][0][4 + i * 6]
        if (confidence > 0.1) {
            val classIndex = outputArray[0][0][4 + i * 6 + 1].toInt()
            val label = if (classIndex >= 0 && classIndex < classes.size) classes[classIndex] else "Unknown"

            val xMin = outputArray[0][0][4 + i * 6 + 2] * image.width
            val yMin = outputArray[0][0][4 + i * 6 + 3] * image.height
            val xMax = outputArray[0][0][4 + i * 6 + 4] * image.width
            val yMax = outputArray[0][0][4 + i * 6 + 5] * image.height

            detections.add(
                Detection(
                    label,
                    xMin,
                    yMin,
                    xMax - xMin,
                    yMax - yMin,
                    confidence
                )
            )
        }
    }
    return detections
}

fun readClasses(context: Context): List<String> {
    val inputStream = context.assets.open("class.txt")
    val reader = BufferedReader(InputStreamReader(inputStream))
    return reader.readLines()
}
