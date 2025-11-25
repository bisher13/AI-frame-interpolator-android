# Add project specific ProGuard rules here.
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-keepclassmembers class * {
    @org.tensorflow.lite.annotations.UsedByReflection *;
}
