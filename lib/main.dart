import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';


void main() {
  runApp(const MaterialApp(
    home: HomeScreen(),
    debugShowCheckedModeBanner: false,
  ));
}

// --- 1. LAYAR MENU UTAMA ---
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("YOLOv11 Flutter App")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.image),
              label: const Text("Input Gambar (Galeri)"),
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(20)),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const ImageDetectionScreen()),
                );
              },
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              icon: const Icon(Icons.camera_alt),
              label: const Text("Buka Kamera (Real-time)"),
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(20)),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const CameraDetectionScreen()),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}

// --- 2. LAYAR DETEKSI GAMBAR (INPUT GAMBAR) ---
class ImageDetectionScreen extends StatefulWidget {
  const ImageDetectionScreen({super.key});

  @override
  State<ImageDetectionScreen> createState() => _ImageDetectionScreenState();
}

class _ImageDetectionScreenState extends State<ImageDetectionScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _image;
  List<dynamic>? _results;
  YOLO? _yolo;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    // Ganti 'yolo11n' dengan nama model Anda (tanpa ekstensi untuk iOS, dengan .tflite untuk Android jika perlu)
    _yolo = YOLO(
      modelPath: 'best_float32.tflite', 
      task: YOLOTask.detect, // Pastikan task sesuai (detect, segment, dll)
    );
    await _yolo!.loadModel();
  }

  Future<void> _pickImage() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      setState(() {
        _image = File(photo.path);
        _isLoading = true;
      });
      _predict();
    }
  }

  Future<void> _predict() async {
    if (_image == null || _yolo == null) return;
    
    final Uint8List imageBytes = await _image!.readAsBytes();
    final result = await _yolo!.predict(imageBytes); //
    
    setState(() {
      // Mengambil daftar kotak pembatas (bounding boxes)
      _results = result['boxes']; 
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Deteksi Gambar")),
      body: Column(
        children: [
          Expanded(
            child: _image == null
                ? const Center(child: Text("Belum ada gambar dipilih"))
                : Stack(
                    children: [
                      Image.file(_image!, fit: BoxFit.contain, width: double.infinity),
                      // Tampilkan overlay hasil deteksi sederhana
                      if (_results != null)
                        ListView.builder(
                          itemCount: _results!.length,
                          itemBuilder: (context, index) {
                            final box = _results![index];
                            return Container(
                              margin: const EdgeInsets.all(4),
                              padding: const EdgeInsets.all(8),
                              color: Colors.black54,
                              child: Text(
                                "${box['class']}: ${(box['confidence'] * 100).toStringAsFixed(1)}%",
                                style: const TextStyle(color: Colors.white),
                              ),
                            );
                          },
                        ),
                    ],
                  ),
          ),
          if (_isLoading) const CircularProgressIndicator(),
          Padding(
            padding: const EdgeInsets.all(100.0),
            child: ElevatedButton(
              onPressed: _pickImage,
              child: const Text("Pilih Gambar"),
            ),
          ),
        ],
      ),
    );
  }
}

// --- 3. LAYAR DETEKSI KAMERA (REAL-TIME) ---
class CameraDetectionScreen extends StatefulWidget {
  const CameraDetectionScreen({super.key});

  @override
  State<CameraDetectionScreen> createState() => _CameraDetectionScreenState();
}

class _CameraDetectionScreenState extends State<CameraDetectionScreen> {
  late YOLOViewController _controller;
  List<dynamic> _currentDetections = [];

  @override
  void initState() {
    super.initState();
    _controller = YOLOViewController();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Kamera Real-time")),
      body: Stack(
        children: [
          // Widget Kamera Bawaan Plugin
          YOLOView(
            modelPath: 'best_float32.tflite', // Ganti dengan nama model Anda
            task: YOLOTask.detect,
            controller: _controller,
            onResult: (results) {
              // Callback ini memberikan hasil deteksi setiap frame
              // Perhatikan: format 'results' di onResult YOLOView mungkin berupa List<YOLOResult> 
              // atau Map tergantung versi. Di versi terbaru, ini seringkali list object.
              setState(() {
                 // Kita simpan untuk menampilkan jumlah objek terdeteksi
                 // Anda bisa print(results) untuk melihat struktur datanya
                 _currentDetections = results;
              });
            },
          ),
          
          // Overlay Informasi Sederhana
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.6),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Text(
                "Objek Terdeteksi: ${_currentDetections.length}",
                style: const TextStyle(color: Colors.white, fontSize: 18),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ],
      ),
    );
  }
}