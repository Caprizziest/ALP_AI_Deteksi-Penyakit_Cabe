import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
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
  ui.Image? _decodedImage; // Untuk menyimpan info ukuran asli gambar

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    _yolo = YOLO(
      modelPath: 'best_float32.tflite',
      task: YOLOTask.detect,
    );
    await _yolo!.loadModel();
  }

  Future<void> _pickImage() async {
    final XFile? photo = await _picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      final file = File(photo.path);
      // Decode gambar untuk mendapatkan ukuran aslinya (penting untuk scaling kotak)
      final data = await file.readAsBytes();
      final decoded = await decodeImageFromList(data);

      setState(() {
        _image = file;
        _decodedImage = decoded;
        _isLoading = true;
        _results = null;
      });
      _predict();
    }
  }

  Future<void> _predict() async {
    if (_image == null || _yolo == null) return;

    final Uint8List imageBytes = await _image!.readAsBytes();
    final result = await _yolo!.predict(imageBytes);

    setState(() {
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
                : LayoutBuilder(
                    builder: (context, constraints) {
                      return Stack(
                        alignment: Alignment.center,
                        children: [
                          // Layer 1: Gambar
                          Image.file(
                            _image!,
                            fit: BoxFit.contain,
                            width: constraints.maxWidth,
                            height: constraints.maxHeight,
                          ),
                          // Layer 2: Bounding Box Overlay
                          if (_results != null && _decodedImage != null)
                            CustomPaint(
                              size: Size(constraints.maxWidth, constraints.maxHeight),
                              painter: BoundingBoxPainter(
                                results: _results!,
                                originalSize: Size(
                                  _decodedImage!.width.toDouble(),
                                  _decodedImage!.height.toDouble(),
                                ),
                                displaySize: Size(
                                  constraints.maxWidth,
                                  constraints.maxHeight,
                                ),
                              ),
                            ),
                        ],
                      );
                    },
                  ),
          ),
          if (_isLoading) const CircularProgressIndicator(),
          Padding(
            padding: const EdgeInsets.all(20.0),
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

// Painter untuk menggambar kotak di atas gambar
class BoundingBoxPainter extends CustomPainter {
  final List<dynamic> results;
  final Size originalSize; // Ukuran asli gambar (misal 1920x1080)
  final Size displaySize;  // Ukuran gambar di layar HP

  BoundingBoxPainter({
    required this.results,
    required this.originalSize,
    required this.displaySize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    final textPaint = TextPainter(
      textDirection: TextDirection.ltr,
    );

    // Hitung faktor scaling agar sesuai dengan BoxFit.contain
    final double scaleX = displaySize.width / originalSize.width;
    final double scaleY = displaySize.height / originalSize.height;
    final double scale = scaleX < scaleY ? scaleX : scaleY;

    // Hitung offset (jarak kosong di kiri/atas karena gambar di-center)
    final double offsetX = (displaySize.width - (originalSize.width * scale)) / 2;
    final double offsetY = (displaySize.height - (originalSize.height * scale)) / 2;

    for (var box in results) {
      // Ambil data dari hasil deteksi
      // Asumsi format box dari plugin: {'x': ..., 'y': ..., 'width': ..., 'height': ...}
      // Koordinat ini biasanya relatif terhadap ukuran asli gambar
      final double x = (box['x'] ?? 0).toDouble();
      final double y = (box['y'] ?? 0).toDouble();
      final double w = (box['width'] ?? 0).toDouble();
      final double h = (box['height'] ?? 0).toDouble();
      final String label = "${box['class']} ${(box['confidence'] * 100).toStringAsFixed(0)}%";

      // Transformasi koordinat ke ukuran layar
      final double left = (x * scale) + offsetX;
      final double top = (y * scale) + offsetY;
      final double width = w * scale;
      final double height = h * scale;

      final rect = Rect.fromLTWH(left, top, width, height);
      canvas.drawRect(rect, paint);

      // Gambar Label Teks
      textPaint.text = TextSpan(
        text: label,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          backgroundColor: Colors.red,
        ),
      );
      textPaint.layout();
      textPaint.paint(canvas, Offset(left, top - 20)); // Gambar teks sedikit di atas kotak
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
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
  bool _isControllerReady = false;

  @override
  void initState() {
    super.initState();
    _controller = YOLOViewController();
    
    // PERBAIKAN: Set threshold langsung di sini sebelum View dibuat.
    // Fungsi ini akan menyimpan nilai setting dan menerapkannya otomatis 
    // begitu kamera/model siap (init).
    _controller.setThresholds(
      confidenceThreshold: 0.2, // 20% (Lebih sensitif)
      iouThreshold: 0.4,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Kamera Real-time")),
      body: Stack(
        children: [
          YOLOView(
            modelPath: 'best_float32.tflite',
            task: YOLOTask.detect,
            controller: _controller,
            // PERBAIKAN: Tambahkan parameter threshold di sini juga agar sinkron saat start
            confidenceThreshold: 0.2, 
            iouThreshold: 0.4,
            onResult: (results) {
              setState(() {
                _currentDetections = results;
                // Jika sudah ada hasil, berarti controller sudah aktif
                if (!_isControllerReady) _isControllerReady = true;
              });
            },
          ),
          
          // Overlay Informasi
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
              child: Column(
                children: [
                  Text(
                    "Objek Terdeteksi: ${_currentDetections.length}",
                    style: const TextStyle(color: Colors.white, fontSize: 18),
                    textAlign: TextAlign.center, 
                  ),
                  // Indikator sederhana
                  if (_currentDetections.isEmpty)
                     const Padding(
                       padding: EdgeInsets.only(top: 8.0),
                       child: Text(
                         "Mencari objek...",
                         style: TextStyle(color: Colors.grey, fontSize: 12),
                       ),
                     )
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}