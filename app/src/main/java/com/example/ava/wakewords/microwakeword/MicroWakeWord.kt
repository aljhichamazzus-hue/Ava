package com.example.ava.wakewords.microwakeword

import com.example.ava.wakewords.models.WakeWordWithId
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import timber.log.Timber
import java.nio.ByteBuffer

class MicroWakeWord(
    val id: String,
    val wakeWord: String,
    model: ByteBuffer,
    private val probabilityCutoff: Float,
    private val slidingWindowSize: Int
) : AutoCloseable {
    private val interpreter: Interpreter = Interpreter(model)
    private val inputTensorBuffer: TensorBuffer
    private val outputScale: Float
    private val outputZeroPoint: Int
    private val outputDataType: DataType  // NEW: Store output data type
    private val probabilities = ArrayDeque<Float>(slidingWindowSize)

    init {
        interpreter.allocateTensors()

        val inputDetails = interpreter.getInputTensor(0)
        val inputQuantParams = inputDetails.quantizationParams()
        inputTensorBuffer = TensorBuffer.create(
            inputDetails.dataType(),
            inputDetails.shape(),
            inputQuantParams.scale,
            inputQuantParams.zeroPoint
        )

        val outputDetails = interpreter.getOutputTensor(0)
        outputDataType = outputDetails.dataType()  // NEW: Get output data type

        val outputQuantParams = outputDetails.quantizationParams()
        outputScale = outputQuantParams.scale
        outputZeroPoint = outputQuantParams.zeroPoint
    }

    fun processAudioFeatures(features: FloatArray): Boolean {
        if (features.isEmpty())
            return false

        inputTensorBuffer.put(features)
        if (!inputTensorBuffer.isComplete)
            return false

        val probability = getWakeWordProbability(inputTensorBuffer.getTensor())
        inputTensorBuffer.clear()
        return isWakeWordDetected(probability)
    }

    private fun getWakeWordProbability(input: ByteBuffer): Float {
        return when (outputDataType) {
            DataType.FLOAT32 -> {
                // Handle FLOAT32 output
                val output = Array(1) { FloatArray(1) }
                interpreter.run(input, output)
                output[0][0]  // Already dequantized
            }
            DataType.UINT8, DataType.INT8 -> {
                // Handle UINT8/INT8 output (original code)
                val output = Array(1) { ByteArray(1) }
                interpreter.run(input, output)
                (output[0][0].toUByte().toFloat() - outputZeroPoint) * outputScale
            }
            else -> {
                throw IllegalArgumentException("Unsupported output data type: $outputDataType")
            }
        }
    }

    private fun isWakeWordDetected(probability: Float): Boolean {
        if (probability > 0.3)
            Timber.d("Probability: $probability")

        if (probabilities.size == slidingWindowSize)
            probabilities.removeFirst()
        probabilities.add(probability)
        return probabilities.size == slidingWindowSize && probabilities.average() > probabilityCutoff
    }

    override fun close() {
        interpreter.close()
    }

    companion object {
        suspend fun fromWakeWord(wakeWord: WakeWordWithId): MicroWakeWord = MicroWakeWord(
            id = wakeWord.id,
            wakeWord = wakeWord.wakeWord.wake_word,
            model = wakeWord.load(),
            probabilityCutoff = wakeWord.wakeWord.micro.probability_cutoff,
            slidingWindowSize = wakeWord.wakeWord.micro.sliding_window_size
        )
    }
}