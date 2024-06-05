package com.example.myapplication

data class Detection(
    val label: String,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val confidence: Float
)
