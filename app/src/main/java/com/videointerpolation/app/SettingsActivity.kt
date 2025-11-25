package com.videointerpolation.app

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import com.videointerpolation.app.data.AppSettings
import com.google.android.material.slider.Slider

class SettingsActivity : AppCompatActivity() {
    
    private lateinit var settings: AppSettings
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        
        settings = AppSettings.getInstance(this)
        
        try {
            setupQualityPresets()
            setupSpeedControls()
            setupExportOptions()
            setupAdvancedSettings()
            
            findViewById<Button>(R.id.btnResetDefaults)?.setOnClickListener {
                settings.resetToDefaults()
                recreate() // Reload UI
            }
            
            findViewById<Button>(R.id.btnSaveSettings)?.setOnClickListener {
                Toast.makeText(this, "Settings saved!", Toast.LENGTH_SHORT).show()
                finish()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "Error loading settings: ${e.message}", Toast.LENGTH_LONG).show()
            android.util.Log.e("SettingsActivity", "Error in onCreate", e)
            finish()
        }
    }
    
    private fun setupQualityPresets() {
        val radioGroup = findViewById<RadioGroup>(R.id.rgQualityPreset) ?: return
        
        when (settings.qualityPreset) {
            AppSettings.QualityPreset.FAST -> radioGroup.check(R.id.rbFast)
            AppSettings.QualityPreset.BALANCED -> radioGroup.check(R.id.rbBalanced)
            AppSettings.QualityPreset.QUALITY -> radioGroup.check(R.id.rbQuality)
        }
        
        radioGroup.setOnCheckedChangeListener { _, checkedId ->
            settings.qualityPreset = when (checkedId) {
                R.id.rbFast -> AppSettings.QualityPreset.FAST
                R.id.rbQuality -> AppSettings.QualityPreset.QUALITY
                else -> AppSettings.QualityPreset.BALANCED
            }
            updateQualityDescription()
        }
        
        updateQualityDescription()
    }
    
    private fun updateQualityDescription() {
        val tvDesc = findViewById<TextView>(R.id.tvQualityDescription) ?: return
        tvDesc.text = when (settings.qualityPreset) {
            AppSettings.QualityPreset.FAST -> 
                "480p processing, fewer interpolated frames\nFastest processing, good for previews"
            AppSettings.QualityPreset.BALANCED -> 
                "720p processing, standard frame count\nGood balance of speed and quality"
            AppSettings.QualityPreset.QUALITY -> 
                "Original resolution, maximum frames\nBest quality, slowest processing"
        }
    }
    
    private fun setupSpeedControls() {
        val spinnerSpeed = findViewById<Spinner>(R.id.spinnerSpeedFactor) ?: return
        val sliderCustomSpeed = findViewById<Slider>(R.id.sliderCustomSpeed) ?: return
        val tvCustomSpeed = findViewById<TextView>(R.id.tvCustomSpeedValue) ?: return
        
        // Speed factor spinner
        val speedOptions = AppSettings.SpeedFactor.values().map { it.displayName }
        spinnerSpeed.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, speedOptions).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinnerSpeed.setSelection(settings.speedFactor.ordinal)
        
        spinnerSpeed.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                settings.speedFactor = AppSettings.SpeedFactor.values()[position]
                sliderCustomSpeed.isEnabled = settings.speedFactor == AppSettings.SpeedFactor.CUSTOM
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // Custom speed slider
        sliderCustomSpeed.value = settings.customSpeedMultiplier.toFloat()
        sliderCustomSpeed.isEnabled = settings.speedFactor == AppSettings.SpeedFactor.CUSTOM
        tvCustomSpeed.text = "${settings.customSpeedMultiplier}x"
        
        sliderCustomSpeed.addOnChangeListener { _, value, _ ->
            settings.customSpeedMultiplier = value.toInt()
            tvCustomSpeed.text = "${value.toInt()}x"
        }
        
        // Target FPS slider
        val sliderFps = findViewById<Slider>(R.id.sliderTargetFps) ?: return
        val tvFps = findViewById<TextView>(R.id.tvTargetFpsValue) ?: return
        sliderFps.value = settings.targetFps.toFloat()
        tvFps.text = "${settings.targetFps} FPS"
        
        sliderFps.addOnChangeListener { _, value, _ ->
            settings.targetFps = value.toInt()
            tvFps.text = "${value.toInt()} FPS"
        }
    }
    
    private fun setupExportOptions() {
        // Codec selection
        val spinnerCodec = findViewById<Spinner>(R.id.spinnerCodec) ?: return
        val codecOptions = AppSettings.ExportCodec.values().map { it.displayName }
        spinnerCodec.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, codecOptions).apply {
            setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        spinnerCodec.setSelection(settings.exportCodec.ordinal)
        
        spinnerCodec.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                settings.exportCodec = AppSettings.ExportCodec.values()[position]
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        
        // Bitrate slider
        val sliderBitrate = findViewById<Slider>(R.id.sliderBitrate) ?: return
        val tvBitrate = findViewById<TextView>(R.id.tvBitrateValue) ?: return
        sliderBitrate.value = settings.exportBitrateMbps
        tvBitrate.text = "${settings.exportBitrateMbps} Mbps"
        
        sliderBitrate.addOnChangeListener { _, value, _ ->
            settings.exportBitrateMbps = value
            tvBitrate.text = "${"%.1f".format(value)} Mbps"
        }
        
        // GIF export checkbox
        val cbExportGif = findViewById<CheckBox>(R.id.cbExportGif) ?: return
        cbExportGif.isChecked = settings.exportAsGif
        cbExportGif.setOnCheckedChangeListener { _, isChecked ->
            settings.exportAsGif = isChecked
        }
    }
    
    private fun setupAdvancedSettings() {
        // Scene detection
        val cbSceneDetection = findViewById<CheckBox>(R.id.cbSceneDetection) ?: return
        cbSceneDetection.isChecked = settings.sceneDetectionEnabled
        cbSceneDetection.setOnCheckedChangeListener { _, isChecked ->
            settings.sceneDetectionEnabled = isChecked
        }
        
        // Frame padding
        val cbFramePadding = findViewById<CheckBox>(R.id.cbFramePadding) ?: return
        cbFramePadding.isChecked = settings.enableFramePadding
        cbFramePadding.setOnCheckedChangeListener { _, isChecked ->
            settings.enableFramePadding = isChecked
        }
        
        // Timelapse mode
        val cbTimelapse = findViewById<CheckBox>(R.id.cbTimelapseMode) ?: return
        cbTimelapse.isChecked = settings.timelapseMode
        cbTimelapse.setOnCheckedChangeListener { _, isChecked ->
            settings.timelapseMode = isChecked
        }
    }
}
